"""Checks that baseline and diversion runs see the SAME patients and service times.
Needs simpy and the repo's simulation package; skipped otherwise.
Before the engine patch this should FAIL once diversion triggers (consult fractions were
drawn from the shared service RNG inside patient_journey, shifting the stream)."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

try:
    from dataclasses import replace, is_dataclass
    from simulation.config import DEFAULT_CONFIG
    from simulation.engine import run_simulation
    HAVE = is_dataclass(DEFAULT_CONFIG)
except Exception:  # simpy or package missing
    HAVE = False


@unittest.skipUnless(HAVE, "simpy / simulation package not available")
class CommonRandomNumbers(unittest.TestCase):
    def test_same_patients_and_service_times_across_scenarios(self):
        fc = pd.DataFrame({"timestamp": pd.date_range("2023-12-23", periods=48, freq="h"),
                           "forecast": 14.0})
        fac = pd.DataFrame({"facility_id": ["F1", "F2"], "facility_name": ["A", "B"],
                            "travel_time_mins": [10, 20], "max_simultaneous_patients": [20, 20]})
        small = dict(num_beds=8, num_doctors=3)
        base = run_simulation(fc, replace(DEFAULT_CONFIG, max_occupancy_before_diversion=None, **small), fac, seed=7)
        div = run_simulation(fc, replace(DEFAULT_CONFIG, max_occupancy_before_diversion=0.5, **small), fac, seed=7)
        if int(div.patients["diversion_accepted"].sum()) == 0:
            self.skipTest("diversion never triggered; raise load in this test")
        cols = ["patient_id", "acuity", "arrival_time", "triage_service", "treatment_time"]
        a = base.patients[cols].sort_values("patient_id").reset_index(drop=True)
        b = div.patients[cols].sort_values("patient_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    unittest.main()