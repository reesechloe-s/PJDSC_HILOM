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


class MemoTests(unittest.TestCase):
    def _facts(self):
        from memo import build_memo_facts
        from kpis import compare_scenarios
        b = {"mean_wait_min": 60.0, "peak_hourly_wait_min": 90.0, "capacity_violation_rate": 0.2}
        o = {"mean_wait_min": 30.0, "peak_hourly_wait_min": 40.0, "capacity_violation_rate": 0.1}
        sim = pd.DataFrame({"utilization": [0.5, 0.97], "saturated": [False, True],
                            "divert_pct": [0.0, 45.5]})
        fac = pd.DataFrame({"facility_name": ["A", "B"], "travel_time_mins": [12, 20],
                            "simultaneous_capacity": [5, 8], "referrals_accepted": [7, 3]})
        fc = pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-23", "2026-10-30"]),
                           "forecast": [10.0, 24.4]})
        params = {"beds": 70, "nurses": 13, "doctors": 4, "trigger_pct": 85}
        return build_memo_facts(params, compare_scenarios(b, o), b, o, sim, fac, 10, fc)

    def test_template_contains_verified_numbers_and_caveats(self):
        from memo import render_template_memo, find_unverified_numbers
        f = self._facts(); t = render_template_memo(f)
        for needle in ("70 beds", "60.0", "30.0", "PASS", "human must review", "24.4"):
            self.assertIn(needle, t)
        self.assertEqual(find_unverified_numbers(t, f), [])

    def test_guard_flags_invented_number(self):
        from memo import find_unverified_numbers
        f = self._facts()
        self.assertEqual(find_unverified_numbers("Wait fell to 30.0 min, saving 123 beds", f), ["123"])

    def test_no_key_uses_template(self):
        from memo import generate_memo
        r = generate_memo(self._facts(), api_key=None)
        self.assertEqual(r.source, "template"); self.assertIn("No API key", r.note)

    def test_llm_success_and_failure_paths(self):
        from memo import generate_memo
        class Resp: text = "Mean wait 30.0 min; 999 patients."
        class Blocked: text = None
        class Ok:
            class models:
                @staticmethod
                def generate_content(**kw): return Resp()
        class Empty:
            class models:
                @staticmethod
                def generate_content(**kw): return Blocked()
        class Boom:
            class models:
                @staticmethod
                def generate_content(**kw): raise TimeoutError()
        ok = generate_memo(self._facts(), api_key="x", _client=Ok())
        self.assertEqual((ok.source, ok.unverified_numbers), ("llm", ["999"]))
        bad = generate_memo(self._facts(), api_key="x", _client=Boom())
        self.assertEqual(bad.source, "template"); self.assertIn("TimeoutError", bad.note)
        blocked = generate_memo(self._facts(), api_key="x", _client=Empty())
        self.assertEqual(blocked.source, "template"); self.assertIn("ValueError", blocked.note)

    def test_prompt_embeds_facts(self):
        from memo import build_llm_prompt
        self.assertIn('"beds": 70', build_llm_prompt(self._facts()))

    def test_template_says_so_when_diversion_never_triggered(self):
        from memo import render_template_memo, find_unverified_numbers
        f = self._facts(); f["patients_diverted"] = 0; f["hours_at_diversion_threshold"] = 0
        txt = render_template_memo(f)
        self.assertIn("Diversion was not triggered", txt)
        self.assertIn("identical", txt)
        self.assertNotIn("Pre-alert", txt)
        self.assertNotIn("\n  ", txt.split("## Findings")[1].split("## Recommended")[0])  # no wrapped lines
        self.assertEqual(find_unverified_numbers(txt, f), [])


class EnvAndStageTests(unittest.TestCase):
    def test_read_env_key_variants(self):
        import tempfile
        from memo import read_env_key
        cases = {
            "GEMINI_API_KEY=abc123\n": "abc123",
            "# c\nGEMINI_API_KEY = \"abc 1\"\n": "abc 1",
            "export GEMINI_API_KEY='k2'  \n": "k2",
            "OTHER=x\nGEMINI_API_KEY=k3 # note\n": "k3",
            "\ufeffGEMINI_API_KEY=bom\n": "bom",
            "GEMINI_API_KEY=\n": None,
            "NOPE=1\n": None,
        }
        for text, want in cases.items():
            with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False, encoding="utf-8") as f:
                f.write(text)
            self.assertEqual(read_env_key(f.name), want, text)
        self.assertIsNone(read_env_key("/nonexistent/.env"))

    def test_stage_breakdown_in_facts_and_memo(self):
        from memo import build_memo_facts, render_template_memo, find_unverified_numbers
        from kpis import compare_scenarios
        b = {"mean_wait_min": 5.0, "peak_hourly_wait_min": 50.0, "capacity_violation_rate": 0.0}
        sim = pd.DataFrame({"utilization": [0.3, 0.2], "saturated": [False, False], "divert_pct": [0.0, 0.0],
                            "mean_triage_wait": [1.0, 2.0], "mean_bed_wait": [0.0, 0.0],
                            "mean_provider_wait": [40.0, 60.0],
                            "mean_provider_utilization": [0.9, 1.0], "mean_nurse_utilization": [0.2, 0.3]})
        fac = pd.DataFrame({"facility_name": ["A"], "travel_time_mins": [10],
                            "max_simultaneous_patients": [5], "referrals_accepted": [0]})
        fc = pd.DataFrame({"timestamp": pd.to_datetime(["2023-12-23", "2023-12-24"]), "forecast": [5.0, 9.0]})
        f = build_memo_facts({"beds": 70, "nurses": 13, "doctors": 4, "trigger_pct": 85},
                             compare_scenarios(b, b), b, b, sim, fac, 0, fc)
        self.assertEqual(f["avg_hourly_wait_by_stage_min"], {"triage": 1.5, "bed": 0.0, "doctor": 50.0})
        self.assertEqual(f["peak_doctor_utilization_pct"], 100.0)
        txt = render_template_memo(f)
        self.assertIn("longest average wait is at the doctor stage", txt)
        self.assertEqual(find_unverified_numbers(txt, f), [])


class ApiKeyLookup(unittest.TestCase):
    def test_priority_env_file_then_secrets_and_both_names(self):
        import tempfile
        from memo import get_api_key
        saved = {k: os.environ.pop(k, None) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
        try:
            self.assertIsNone(get_api_key([], None))
            with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
                f.write("GOOGLE_API_KEY=from_file\n")
            self.assertEqual(get_api_key([f.name], {"GEMINI_API_KEY": "s"}), "from_file")
            self.assertEqual(get_api_key(["/nope/.env"], {"GEMINI_API_KEY": "s"}), "s")
            os.environ["GEMINI_API_KEY"] = "from_env"
            self.assertEqual(get_api_key([f.name], {"GEMINI_API_KEY": "s"}), "from_env")
        finally:
            os.environ.pop("GEMINI_API_KEY", None)
            for k, v in saved.items():
                if v is not None: os.environ[k] = v


class LlmResilience(unittest.TestCase):
    def setUp(self):
        import memo
        memo.RETRY_DELAY_S = 0.0
        self.memo = memo
        self.facts = MemoTests()._facts()

    def _client(self, behaviours):
        calls = []
        class R:  # response
            def __init__(s, text): s.text = text
        class C:
            class models:
                @staticmethod
                def generate_content(model, contents, config):
                    calls.append(model)
                    b = behaviours.pop(0)
                    if isinstance(b, Exception): raise b
                    return R(b)
        return C(), calls

    def test_retries_once_on_5xx_then_succeeds(self):
        class ServerError(Exception): code = 503
        c, calls = self._client([ServerError("503 UNAVAILABLE overloaded"), "Mean wait 3.9 min"])
        r = self.memo.generate_memo(self.facts, api_key="k", model="primary", _client=c)
        self.assertEqual((r.source, calls, r.note), ("llm", ["primary", "primary"], ""))

    def test_falls_back_to_second_model_and_says_so(self):
        class ClientError(Exception): code = 404
        c, calls = self._client([ClientError("404 NOT_FOUND model"), "Wait 30.0 min"])
        r = self.memo.generate_memo(self.facts, api_key="k", model="bad-name", _client=c)
        self.assertEqual(r.source, "llm"); self.assertEqual(calls[0], "bad-name")
        self.assertNotEqual(calls[1], "bad-name"); self.assertIn("used", r.note)

    def test_error_message_shown_and_key_redacted(self):
        class ServerError(Exception): code = 503
        e = ServerError("503 UNAVAILABLE key=SECRET123 overloaded")
        c, _ = self._client([e, e, e, e])
        r = self.memo.generate_memo(self.facts, api_key="SECRET123", model="p", _client=c)
        self.assertEqual(r.source, "template")
        self.assertIn("503 UNAVAILABLE", r.note); self.assertNotIn("SECRET123", r.note)