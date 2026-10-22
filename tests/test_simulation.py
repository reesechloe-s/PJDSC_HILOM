import unittest

import pandas as pd

from simulation.config import DEFAULT_CONFIG, SimulationConfig
from simulation.engine3 import run_simulation


class SimulationTests(unittest.TestCase):
    @staticmethod
    def forecast(values: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2026-01-01", periods=len(values), freq="h"
                ),
                "forecast": values,
            }
        )

    def test_run_is_reproducible_for_same_seed(self) -> None:
        config = SimulationConfig(
            num_doctors=2,
            num_nurses=2,
            num_beds=3,
            triage_min_minutes=1,
            triage_max_minutes=1,
            treatment_mean_minutes=2,
            treatment_std_dev_minutes=0,
            diversion_occupancy_threshold=None,
        )

        first = run_simulation(self.forecast([8, 8]), config=config, seed=7)
        second = run_simulation(self.forecast([8, 8]), config=config, seed=7)

        pd.testing.assert_frame_equal(first.patients, second.patients)
        pd.testing.assert_frame_equal(first.hourly, second.hourly)

    def test_default_resource_baseline_runs_without_diversion(self) -> None:
        baseline_config = SimulationConfig(diversion_occupancy_threshold=None)

        result = run_simulation(
            self.forecast([8]),
            config=baseline_config,
            seed=7,
        )

        self.assertEqual(baseline_config.num_beds, DEFAULT_CONFIG.num_beds)
        self.assertEqual(baseline_config.num_nurses, DEFAULT_CONFIG.num_nurses)
        self.assertEqual(baseline_config.num_doctors, DEFAULT_CONFIG.num_doctors)
        self.assertFalse(result.patients["diversion_offered"].any())
        self.assertFalse(result.patients["diversion_accepted"].any())
        self.assertEqual(result.hourly.loc[0, "arrival_count"], len(result.patients))
        self.assertIn("mean_total_wait", result.hourly.columns)

    def test_diversion_only_accepts_nonurgent_and_respects_capacity(self) -> None:
        config = SimulationConfig(
            num_doctors=1,
            num_nurses=1,
            num_beds=1,
            acuity_probabilities={"non-urgent": 1.0},
            triage_min_minutes=0,
            triage_max_minutes=0,
            treatment_mean_minutes=120,
            treatment_std_dev_minutes=0,
            diversion_occupancy_threshold=0.0,
        )
        facilities = pd.DataFrame(
            {
                "facility_id": ["A"],
                "facility_name": ["Facility A"],
                "travel_time_mins": [1],
                "max_simultaneous_patients": [1],
            }
        )

        result = run_simulation(self.forecast([20]), config, facilities, seed=11)

        self.assertTrue(result.patients["diversion_accepted"].any())
        self.assertLessEqual(result.facilities.loc[0, "referrals_accepted"], 1)
        diverted = result.patients.loc[
            result.patients["diversion_accepted"], "acuity"
        ]
        self.assertTrue((diverted == "non-urgent").all())

    def test_scenarios_use_the_same_arrivals_and_service_draws(self) -> None:
        config = SimulationConfig(
            num_doctors=2,
            num_nurses=2,
            num_beds=3,
            triage_min_minutes=1,
            triage_max_minutes=2,
            treatment_mean_minutes=3,
            treatment_std_dev_minutes=0.5,
            diversion_occupancy_threshold=None,
        )
        facilities = pd.DataFrame(
            {
                "facility_id": ["A"],
                "facility_name": ["Facility A"],
                "travel_time_mins": [1],
                "max_simultaneous_patients": [2],
            }
        )
        baseline = run_simulation(self.forecast([12, 12]), config, facilities, seed=19)
        diversion = run_simulation(
            self.forecast([12, 12]),
            SimulationConfig(
                num_doctors=2,
                num_nurses=2,
                num_beds=3,
                triage_min_minutes=1,
                triage_max_minutes=2,
                treatment_mean_minutes=3,
                treatment_std_dev_minutes=0.5,
                diversion_occupancy_threshold=0.0,
            ),
            facilities,
            seed=19,
        )

        matching_columns = [
            "patient_id",
            "acuity",
            "arrival_time",
            "triage_service",
            "treatment_time",
        ]
        pd.testing.assert_frame_equal(
            baseline.patients[matching_columns],
            diversion.patients[matching_columns],
        )

    def test_zero_capacity_facility_rejects_referral_without_breaking_run(self) -> None:
        config = SimulationConfig(
            acuity_probabilities={"non-urgent": 1.0},
            triage_min_minutes=0,
            triage_max_minutes=0,
            treatment_mean_minutes=1,
            treatment_std_dev_minutes=0,
            diversion_occupancy_threshold=0.0,
        )
        facilities = pd.DataFrame(
            {
                "facility_id": ["closed"],
                "facility_name": ["Closed facility"],
                "travel_time_mins": [0],
                "max_simultaneous_patients": [0],
            }
        )

        result = run_simulation(self.forecast([1]), config, facilities, seed=3)

        self.assertEqual(result.facilities.loc[0, "referrals_accepted"], 0)
        self.assertFalse(result.patients["diversion_accepted"].any())
        self.assertTrue(
            (result.patients["diversion_rejection_reason"] == "no_receiving_capacity").all()
        )

    def test_forecast_requires_consecutive_hourly_timestamps(self) -> None:
        forecast = self.forecast([1, 1])
        forecast.loc[1, "timestamp"] += pd.Timedelta(minutes=30)

        with self.assertRaisesRegex(ValueError, "consecutive hourly"):
            run_simulation(forecast)

    def test_invalid_capacity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_beds"):
            SimulationConfig(num_beds=0)


if __name__ == "__main__":
    unittest.main()
