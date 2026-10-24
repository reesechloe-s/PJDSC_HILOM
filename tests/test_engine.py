import unittest

import pandas as pd

from simulation.config import SimulationConfig
from simulation.engine import run_simulation


class Engine3Tests(unittest.TestCase):
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
            max_occupancy_before_diversion=None,
        )

        first = run_simulation(self.forecast([8, 8]), config=config, seed=7)
        second = run_simulation(self.forecast([8, 8]), config=config, seed=7)

        pd.testing.assert_frame_equal(first.patients, second.patients)
        pd.testing.assert_frame_equal(first.hourly, second.hourly)

    def test_treatment_times_use_configured_lognormal_mean(self) -> None:
        config = SimulationConfig(
            num_doctors=2000,
            num_nurses=2000,
            num_beds=2000,
            triage_min_minutes=0,
            triage_max_minutes=0,
            treatment_mean_minutes=42,
            treatment_std_dev_minutes=15,
            max_occupancy_before_diversion=None,
        )

        result = run_simulation(self.forecast([2000]), config=config, seed=12)
        treatment_times = result.patients["treatment_time"]

        self.assertTrue((treatment_times > 0).all())
        self.assertAlmostEqual(treatment_times.mean(), 42, delta=1.5)

    def test_only_nonurgent_patients_are_diverted(self) -> None:
        config = SimulationConfig(
            num_doctors=1,
            num_nurses=1,
            num_beds=1,
            acuity_probabilities={"non-urgent": 1.0},
            triage_min_minutes=0,
            triage_max_minutes=0,
            treatment_mean_minutes=120,
            treatment_std_dev_minutes=0,
            max_occupancy_before_diversion=0.0,
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
        diverted = result.patients.loc[result.patients["diversion_accepted"]]

        self.assertFalse(diverted.empty)
        self.assertTrue((diverted["acuity"] == "non-urgent").all())
        self.assertTrue((diverted["facility_id"] == "A").all())
        self.assertEqual(
            result.facilities.loc[0, "simultaneous_capacity"], 1
        )

    def test_zero_capacity_facility_rejects_referral(self) -> None:
        config = SimulationConfig(
            acuity_probabilities={"non-urgent": 1.0},
            triage_min_minutes=0,
            triage_max_minutes=0,
            treatment_mean_minutes=1,
            treatment_std_dev_minutes=0,
            max_occupancy_before_diversion=0.0,
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
            (
                result.patients["diversion_rejection_reason"]
                == "no_receiving_capacity"
            ).all()
        )

    def test_invalid_facility_data_is_rejected(self) -> None:
        facilities = pd.DataFrame(
            {
                "facility_id": ["A"],
                "facility_name": ["Facility A"],
                "travel_time_mins": [-1],
                "max_simultaneous_patients": [1],
            }
        )
        with self.assertRaisesRegex(ValueError, "travel times"):
            run_simulation(self.forecast([1]), facilities=facilities)

    def test_treatment_standard_deviation_requires_positive_mean(self) -> None:
        config = SimulationConfig(
            treatment_mean_minutes=0,
            treatment_std_dev_minutes=1,
        )

        with self.assertRaisesRegex(ValueError, "positive mean"):
            run_simulation(self.forecast([1]), config=config)


if __name__ == "__main__":
    unittest.main()
