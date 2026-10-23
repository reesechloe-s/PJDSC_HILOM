import math, os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from queueing_theory import erlang_c, mmc_metrics, effective_arrival_rate, servers_needed, mmc_table
from kpis import scenario_kpis, compare_scenarios, forecast_scorecard, seasonal_naive


class QueueTheory(unittest.TestCase):
    def test_mm1_matches_closed_form(self):
        m = mmc_metrics(30, 1.5, 1)                    # rho = 0.75
        self.assertAlmostEqual(m["p_wait"], 0.75)
        self.assertAlmostEqual(m["mean_wait_min"], 0.75 / (1/1.5 - 0.5) , places=6) if False else None
        # Wq = rho/(mu-lam) ; mu=40/h, lam=30/h -> 0.075 h = 4.5 min
        self.assertAlmostEqual(m["mean_wait_min"], 4.5, places=6)

    def test_erlang_c_known_value(self):
        self.assertAlmostEqual(erlang_c(2, 1.0), 1/3)

    def test_unstable_is_infinite_not_negative(self):
        m = mmc_metrics(100, 42, 10)
        self.assertFalse(m["stable"]); self.assertTrue(math.isinf(m["mean_wait_min"]))

    def test_zero_arrivals(self):
        self.assertEqual(mmc_metrics(0, 42, 5)["mean_wait_min"], 0.0)

    def test_large_c_no_overflow(self):
        self.assertTrue(0 <= erlang_c(400, 380) <= 1)

    def test_more_servers_never_hurts(self):
        w = [mmc_metrics(40, 30, c)["mean_wait_min"] for c in range(21, 40)]
        self.assertTrue(all(a >= b for a, b in zip(w, w[1:])))

    def test_diversion_monotone(self):
        base = mmc_metrics(16, 240, 70)["mean_wait_min"]
        div = mmc_metrics(effective_arrival_rate(16, .4, 1), 240, 70)["mean_wait_min"]
        self.assertLess(div, base)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError): effective_arrival_rate(10, 1.5)
        with self.assertRaises(ValueError): mmc_metrics(10, 0, 3)

    def test_servers_needed_meets_target(self):
        c = servers_needed(40, 30, 5)
        self.assertLessEqual(mmc_metrics(40, 30, c)["mean_wait_min"], 5)
        self.assertGreater(mmc_metrics(40, 30, c - 1)["mean_wait_min"], 5)

    def test_table(self):
        t = mmc_table(pd.Series([5, 10, 15], index=[0, 1, 2]), 30, 10)
        self.assertEqual(len(t), 3)


class Kpis(unittest.TestCase):
    def _h(self, waits, utils):
        return pd.DataFrame({"wait_min": waits, "utilization": utils})

    def test_scorecard_pass_and_fail(self):
        b = scenario_kpis(self._h([60, 60, 60, 60], [1, 1, 1, .5]))
        o = scenario_kpis(self._h([30, 30, 30, 30], [1, .9, .9, .5]))
        df = compare_scenarios(b, o)
        self.assertEqual(list(df["Status"]), ["PASS", "PASS", "PASS"])
        worse = compare_scenarios(b, b)
        self.assertEqual(worse["Status"][0], "FAIL")

    def test_zero_baseline_is_not_a_pass(self):
        b = scenario_kpis(self._h([0, 0], [.2, .2]))
        df = compare_scenarios(b, b)
        self.assertTrue(df["Status"][0].startswith("n/a"))

    def test_patient_level_wait_preferred(self):
        h = self._h([10, 10], [.5, .5])
        p = pd.DataFrame({"total_wait": [0, 100]})
        self.assertEqual(scenario_kpis(h, p)["mean_wait_min"], 50.0)

    def test_forecast_metrics(self):
        m = forecast_scorecard([10, 0, 20], [8, 1, 20])
        self.assertAlmostEqual(m["MAE"], 1.0)
        self.assertEqual(m["MAPE_n_excluded_zero_hours"], 1)
        self.assertAlmostEqual(m["WAPE"], 3 / 30)

    def test_seasonal_naive(self):
        s = pd.Series(np.arange(336))
        self.assertEqual(seasonal_naive(s, 3)[0], 168)
        with self.assertRaises(ValueError): seasonal_naive(s.iloc[:10], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class DashboardMetrics(unittest.TestCase):
    def test_divert_pct_and_nan_handling(self):
        from kpis import hourly_dashboard_metrics
        hourly = pd.DataFrame({"mean_bed_utilization": [0.5, np.nan, 0.9],
                               "mean_total_wait": [5.0, np.nan, 50.0]}, index=[10, 11, 12])
        pts = pd.DataFrame({"arrival_time": [5, 10, 70, 80, 90],
                            "acuity": ["non-urgent"] * 3 + ["urgent", "non-urgent"],
                            "diversion_accepted": [True, False, True, True, True]})
        out = hourly_dashboard_metrics(hourly, pts, 0.85)
        self.assertEqual(list(out["divert_pct"]), [50.0, 100.0, 0.0])
        self.assertEqual(list(out["saturated"]), [False, False, True])
        self.assertEqual(out["wait_min"][1], 0.0)

    def test_empty_or_missing_patients(self):
        from kpis import hourly_dashboard_metrics
        h = pd.DataFrame({"mean_bed_utilization": [.5], "mean_total_wait": [1.0]})
        for p in (None, pd.DataFrame()):
            self.assertEqual(hourly_dashboard_metrics(h, p, .85)["divert_pct"][0], 0.0)

    def test_out_of_range_hours_ignored(self):
        from kpis import hourly_dashboard_metrics
        h = pd.DataFrame({"mean_bed_utilization": [.5], "mean_total_wait": [1.0]})
        p = pd.DataFrame({"arrival_time": [9999], "acuity": ["non-urgent"], "diversion_accepted": [True]})
        self.assertEqual(hourly_dashboard_metrics(h, p, .85)["divert_pct"][0], 0.0)