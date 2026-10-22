# includes the queuing logic

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import simpy

from simulation.config import SimulationConfig

ACUITY_PRIORITY = {"emergent": 0, "urgent": 1, "non-urgent": 2}
PATIENT_COLUMNS = [
    "patient_id",
    "acuity",
    "arrival_time",
    "triage_wait",
    "triage_service",
    "bed_wait",
    "provider_wait",
    "treatment_time",
    "travel_time",
    "total_wait",
    "time_in_system",
    "status",
    "diversion_offered",
    "diversion_accepted",
    "diversion_rejection_reason",
    "facility_id",
    "facility_name",
]
FACILITY_COLUMNS = [
    "facility_id",
    "facility_name",
    "referrals_accepted",
    "simultaneous_capacity",
]


@dataclass
class SimulationResult:
    """Patient-level outcomes and hourly/facility summaries for one run."""

    patients: pd.DataFrame
    hourly: pd.DataFrame
    facilities: pd.DataFrame


class _EmergencyDepartment:
    def __init__(
        self,
        env: simpy.Environment,
        config: SimulationConfig,
        rng: np.random.Generator,
        facility_data: pd.DataFrame,
    ) -> None:
        self.env = env
        self.config = config
        self.rng = rng
        self.triage_nurses = simpy.Resource(env, capacity=config.num_nurses)
        self.beds = simpy.PriorityResource(env, capacity=config.num_beds)
        self.providers = simpy.PriorityResource(env, capacity=config.num_doctors)
        self.facilities = facility_data.to_dict(orient="records")
        self.facility_resources = {
            facility["facility_id"]: simpy.Resource(
                env, capacity=int(facility["max_simultaneous_patients"])
            )
            for facility in self.facilities
            if int(facility["max_simultaneous_patients"]) > 0
        }
        self.patient_records: list[dict[str, Any]] = []
        self.accepted_referrals = {facility["facility_id"]: 0 for facility in self.facilities}

    def _service_time(self, *, triage: bool) -> float:
        if triage:
            return float(
                self.rng.uniform(
                    self.config.triage_min_minutes,
                    self.config.triage_max_minutes,
                )
            )
        if self.config.treatment_min_minutes == self.config.treatment_max_minutes:
            return self.config.treatment_min_minutes
        return float(
            self.rng.triangular(
                self.config.treatment_min_minutes,
                self.config.treatment_mode_minutes,
                self.config.treatment_max_minutes,
            )
        )

    def _bed_pressure(self) -> float:
        return (self.beds.count + len(self.beds.queue)) / self.config.num_beds

    def _available_facility(self) -> dict[str, Any] | None:
        for facility in sorted(
            self.facilities, key=lambda item: float(item["travel_time_mins"])
        ):
            if int(facility["max_simultaneous_patients"]) == 0:
                continue
            resource = self.facility_resources[facility["facility_id"]]
            if resource.count + len(resource.queue) < resource.capacity:
                return facility
        return None

    def _attempt_diversion(self, record: dict[str, Any]):
        record["diversion_offered"] = True
        facility = self._available_facility()
        if facility is None:
            record["diversion_rejection_reason"] = "no_receiving_capacity"
            return False

        resource = self.facility_resources[facility["facility_id"]]
        with resource.request() as request:
            yield request
            record["facility_id"] = facility["facility_id"]
            record["facility_name"] = facility["facility_name"]
            record["travel_time"] = float(facility["travel_time_mins"])
            self.accepted_referrals[facility["facility_id"]] += 1
            record["diversion_accepted"] = True
            record["status"] = "diverted"

            yield self.env.timeout(record["travel_time"])
            yield self.env.timeout(record["treatment_time"])
            record["time_in_system"] = self.env.now - record["arrival_time"]
        return True

    def patient_journey(
        self,
        patient_id: int,
        acuity: str,
        triage_service: float,
        treatment_service: float,
    ):
        arrival_time = float(self.env.now)
        record: dict[str, Any] = {
            "patient_id": patient_id,
            "acuity": acuity,
            "arrival_time": arrival_time,
            "triage_wait": 0.0,
            "triage_service": 0.0,
            "bed_wait": 0.0,
            "provider_wait": 0.0,
            "treatment_time": 0.0,
            "travel_time": 0.0,
            "total_wait": 0.0,
            "time_in_system": 0.0,
            "status": "treated",
            "diversion_offered": False,
            "diversion_accepted": False,
            "diversion_rejection_reason": None,
            "facility_id": None,
            "facility_name": None,
        }
        record["triage_service"] = triage_service
        record["treatment_time"] = treatment_service
        self.patient_records.append(record)

        with self.triage_nurses.request() as nurse_request:
            yield nurse_request
            record["triage_wait"] = self.env.now - arrival_time
            yield self.env.timeout(record["triage_service"])

        threshold = self.config.diversion_occupancy_threshold
        eligible_for_diversion = acuity == "non-urgent"
        if (
            eligible_for_diversion
            and threshold is not None
            and self._bed_pressure() >= threshold
        ):
            accepted = yield from self._attempt_diversion(record)
            if accepted:
                record["total_wait"] = record["triage_wait"]
                return

        priority = ACUITY_PRIORITY.get(acuity, len(ACUITY_PRIORITY))
        bed_request_time = self.env.now
        with self.beds.request(priority=priority) as bed_request:
            yield bed_request
            record["bed_wait"] = self.env.now - bed_request_time

            provider_request_time = self.env.now
            with self.providers.request(priority=priority) as provider_request:
                yield provider_request
                record["provider_wait"] = self.env.now - provider_request_time
                yield self.env.timeout(record["treatment_time"])

        record["total_wait"] = (
            record["triage_wait"] + record["bed_wait"] + record["provider_wait"]
        )
        record["time_in_system"] = self.env.now - arrival_time


def _validate_forecast(forecast_df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "forecast"}
    missing = required.difference(forecast_df.columns)
    if missing:
        raise ValueError(f"Forecast is missing required columns: {sorted(missing)}")
    if forecast_df.empty:
        raise ValueError("Forecast must contain at least one hourly row.")

    forecast = forecast_df.loc[:, ["timestamp", "forecast"]].copy()
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], errors="raise")
    if forecast["timestamp"].isna().any():
        raise ValueError("Forecast timestamps cannot be missing.")
    forecast["forecast"] = pd.to_numeric(forecast["forecast"], errors="raise")
    values = forecast["forecast"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Forecast values must be finite, nonnegative arrivals per hour.")
    if forecast["timestamp"].duplicated().any():
        raise ValueError("Forecast timestamps must be unique.")
    forecast = forecast.sort_values("timestamp", kind="stable").reset_index(drop=True)
    differences = forecast["timestamp"].diff().dropna()
    if not differences.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Forecast timestamps must be consecutive hourly intervals.")
    return forecast


def _validate_facilities(facilities: pd.DataFrame | None) -> pd.DataFrame:
    if facilities is None:
        return pd.DataFrame(
            columns=[
                "facility_id",
                "facility_name",
                "travel_time_mins",
                "max_simultaneous_patients",
            ]
        )

    required = {
        "facility_id",
        "facility_name",
        "travel_time_mins",
        "max_simultaneous_patients",
    }
    missing = required.difference(facilities.columns)
    if missing:
        raise ValueError(f"Facility data is missing required columns: {sorted(missing)}")
    result = facilities.loc[:, sorted(required)].copy()
    if result["facility_id"].isna().any() or result["facility_id"].duplicated().any():
        raise ValueError("Facility IDs must be present and unique.")
    result["travel_time_mins"] = pd.to_numeric(
        result["travel_time_mins"], errors="raise"
    )
    result["max_simultaneous_patients"] = pd.to_numeric(
        result["max_simultaneous_patients"], errors="raise"
    )
    travel_times = result["travel_time_mins"].to_numpy(dtype=float)
    capacities = result["max_simultaneous_patients"].to_numpy(dtype=float)
    if not np.isfinite(travel_times).all() or (travel_times < 0).any():
        raise ValueError("Facility travel times must be finite and nonnegative.")
    if (
        not np.isfinite(capacities).all()
        or (capacities < 0).any()
        or not np.equal(capacities, np.floor(capacities)).all()
    ):
        raise ValueError("Facility capacities must be nonnegative whole numbers.")
    result["max_simultaneous_patients"] = result[
        "max_simultaneous_patients"
    ].astype(int)
    return result.reset_index(drop=True)


def _hourly_results(
    forecast: pd.DataFrame,
    patients: pd.DataFrame,
    samples: list[dict[str, float]],
    config: SimulationConfig,
) -> pd.DataFrame:
    hourly = forecast.copy()
    hourly["arrival_count"] = 0
    hourly["treated_count"] = 0
    hourly["diverted_count"] = 0
    hourly["diversion_offered_count"] = 0
    hourly["diversion_rejected_count"] = 0
    hourly["mean_triage_wait"] = np.nan
    hourly["mean_bed_wait"] = np.nan
    hourly["mean_provider_wait"] = np.nan
    hourly["mean_total_wait"] = np.nan
    hourly["mean_time_in_system"] = np.nan
    hourly["wait_target_exceeded_count"] = 0
    hourly["mean_bed_utilization"] = np.nan
    hourly["mean_nurse_utilization"] = np.nan
    hourly["mean_provider_utilization"] = np.nan

    if not patients.empty:
        patient_frame = patients.copy()
        patient_frame["_hour"] = (
            patient_frame["arrival_time"].floordiv(60).astype(int)
        )
        for hour, group in patient_frame.groupby("_hour"):
            if hour >= len(hourly):
                continue
            hourly.loc[hour, "arrival_count"] = len(group)
            hourly.loc[hour, "treated_count"] = int((group["status"] == "treated").sum())
            hourly.loc[hour, "diverted_count"] = int(
                (group["status"] == "diverted").sum()
            )
            hourly.loc[hour, "diversion_offered_count"] = int(
                group["diversion_offered"].sum()
            )
            hourly.loc[hour, "diversion_rejected_count"] = int(
                group["diversion_rejection_reason"].notna().sum()
            )
            hourly.loc[hour, "wait_target_exceeded_count"] = int(
                (group["total_wait"] > config.wait_target_minutes).sum()
            )
            for metric, source in (
                ("mean_triage_wait", "triage_wait"),
                ("mean_bed_wait", "bed_wait"),
                ("mean_provider_wait", "provider_wait"),
                ("mean_total_wait", "total_wait"),
                ("mean_time_in_system", "time_in_system"),
            ):
                hourly.loc[hour, metric] = group[source].mean()

    if samples:
        sample_frame = pd.DataFrame(samples)
        sample_frame["_hour"] = sample_frame["time"].floordiv(60).astype(int)
        for hour, group in sample_frame.groupby("_hour"):
            if hour >= len(hourly):
                continue
            hourly.loc[hour, "mean_bed_utilization"] = (
                group["beds_in_use"].mean() / config.num_beds
            )
            hourly.loc[hour, "mean_nurse_utilization"] = (
                group["nurses_in_use"].mean() / config.num_nurses
            )
            hourly.loc[hour, "mean_provider_utilization"] = (
                group["providers_in_use"].mean() / config.num_doctors
            )
    return hourly


def run_simulation(
    forecast_df: pd.DataFrame,
    config: SimulationConfig | None = None,
    facilities: pd.DataFrame | None = None,
    seed: int = 42,
) -> SimulationResult:
    """Run a forecast-driven ED simulation and return patient/hour/facility results.

    ``forecast_df`` must have one row per consecutive hour with ``timestamp`` and
    ``forecast`` (expected arrivals in that hour). Service-time distributions,
    arrival sampling, and diversion thresholds are scenario assumptions until
    calibrated against operational data.
    """
    config = config or SimulationConfig()
    forecast = _validate_forecast(forecast_df)
    facility_data = _validate_facilities(facilities)
    arrival_seed, service_seed = np.random.SeedSequence(seed).spawn(2)
    arrival_rng = np.random.default_rng(arrival_seed)
    service_rng = np.random.default_rng(service_seed)
    env = simpy.Environment()
    ed = _EmergencyDepartment(env, config, service_rng, facility_data)
    samples: list[dict[str, float]] = []
    horizon_minutes = len(forecast) * 60.0

    def arrivals():
        patient_id = 0
        probabilities = list(config.acuity_probabilities.values())
        acuities = list(config.acuity_probabilities)

        for row in forecast.itertuples(index=False):
            hour_start = float(env.now)
            count = int(arrival_rng.poisson(float(row.forecast)))
            arrival_offsets = np.sort(arrival_rng.uniform(0.0, 60.0, count))
            for offset in arrival_offsets:
                yield env.timeout(hour_start + float(offset) - env.now)
                acuity = str(arrival_rng.choice(acuities, p=probabilities))
                triage_service = ed._service_time(triage=True)
                treatment_service = ed._service_time(triage=False)
                env.process(
                    ed.patient_journey(
                        patient_id,
                        acuity,
                        triage_service,
                        treatment_service,
                    )
                )
                patient_id += 1
            yield env.timeout(hour_start + 60.0 - env.now)

    def sample_resources():
        sample_interval = 5.0
        while env.now < horizon_minutes:
            yield env.timeout(sample_interval)
            if env.now <= horizon_minutes:
                samples.append(
                    {
                        "time": float(env.now),
                        "beds_in_use": float(ed.beds.count),
                        "nurses_in_use": float(ed.triage_nurses.count),
                        "providers_in_use": float(ed.providers.count),
                    }
                )

    env.process(arrivals())
    env.process(sample_resources())
    env.run()

    patients = pd.DataFrame(ed.patient_records, columns=PATIENT_COLUMNS)
    hourly = _hourly_results(forecast, patients, samples, config)
    facility_results = pd.DataFrame(
        [
            {
                "facility_id": facility["facility_id"],
                "facility_name": facility["facility_name"],
                "referrals_accepted": ed.accepted_referrals[facility["facility_id"]],
                "simultaneous_capacity": int(facility["max_simultaneous_patients"]),
            }
            for facility in ed.facilities
        ],
        columns=FACILITY_COLUMNS,
    )
    return SimulationResult(
        patients=patients,
        hourly=hourly,
        facilities=facility_results,
    )
