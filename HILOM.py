"""Policy-memo generation for the dashboard's 'Policy Memo' tab.

Design rules (from the proposal: LLM drafts from *verified* model outputs, human reviews):
  1. Every number comes from `build_memo_facts`, i.e. from the simulation, never from the LLM.
  2. A deterministic template memo always works (no key, no network, instant).
  3. If an API key is set, an LLM rewrites the same facts; any number it uses that is not
     in the facts is flagged for the reviewer.
No Streamlit imports here, so the module is unit-testable.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field

import pandas as pd

DEFAULT_MODEL = os.environ.get("HILOM_LLM_MODEL", "claude-sonnet-5-5")
LLM_TIMEOUT_S = 10.0  # proposal target: memo within 10 seconds

SYSTEM_PROMPT = (
    "You draft internal triage-protocol memoranda for hospital administrators and local "
    "government officials in the Philippines. Use ONLY the numbers in the JSON facts you are "
    "given; never invent, estimate or round to new figures. State that results come from a "
    "simulation on synthetic data, that the tool is decision support only, that high-acuity "
    "and resuscitation cases are excluded from diversion, and that a human must review before "
    "any action. Where a target status is FAIL or n/a, say so plainly. Plain markdown, under "
    "350 words, with sections: Summary, Findings, Recommended actions, Limitations."
)


@dataclass
class MemoResult:
    text: str
    source: str                      # "llm" or "template"
    note: str = ""                   # why the template was used, if it was
    unverified_numbers: list = field(default_factory=list)


# ----------------------------------------------------------------------------- facts
def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if hasattr(v, "item"):
        v = v.item()
    return v


def build_memo_facts(
    params: dict,
    kpi_table: pd.DataFrame,
    baseline_kpis: dict,
    optimized_kpis: dict,
    sim_df: pd.DataFrame,
    facility_results: pd.DataFrame,
    diverted_total: int,
    forecast_df: pd.DataFrame,
) -> dict:
    fac_cols = [c for c in ("facility_name", "travel_time_mins", "simultaneous_capacity",
                            "referrals_accepted") if c in facility_results.columns]
    facilities = (
        facility_results[fac_cols].sort_values("referrals_accepted", ascending=False)
        if "referrals_accepted" in fac_cols else facility_results[fac_cols]
    ).head(5)
    return {
        "settings": {k: _clean(v) for k, v in params.items()},
        "forecast_window": {
            "start": str(forecast_df["timestamp"].min()),
            "end": str(forecast_df["timestamp"].max()),
            "peak_forecast_arrivals_per_hour": round(float(forecast_df["forecast"].max()), 1),
        },
        "baseline": {k: _clean(v) for k, v in baseline_kpis.items()},
        "with_diversion": {k: _clean(v) for k, v in optimized_kpis.items()},
        "targets": [{k: _clean(v) for k, v in r.items()} for r in kpi_table.to_dict("records")],
        "peak_bed_utilization_pct": round(float(sim_df["utilization"].max()) * 100, 1),
        "hours_at_diversion_threshold": int(sim_df["saturated"].sum()),
        "peak_nonurgent_diversion_pct": round(float(sim_df["divert_pct"].max()), 1),
        "patients_diverted": int(diverted_total),
        "top_receiving_facilities": [
            {k: _clean(v) for k, v in r.items()} for r in facilities.to_dict("records")
        ],
    }


# ------------------------------------------------------------------- template memo
def _pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def render_template_memo(facts: dict) -> str:
    s, b, w = facts["settings"], facts["baseline"], facts["with_diversion"]
    t = {row["KPI"]: row for row in facts["targets"]}
    wait_row = t.get("Mean wait reduction vs baseline", {})
    viol_row = t.get("Capacity-violation reduction vs baseline", {})
    peak_row = next((r for k, r in t.items() if k.startswith("Peak hourly")), {})
    fac_lines = "\n".join(
        f"- {f.get('facility_name', '?')}: {f.get('referrals_accepted', 0)} referrals "
        f"accepted, {f.get('travel_time_mins', '?')} min travel"
        for f in facts["top_receiving_facilities"]
    ) or "- No facility data available."
    return f"""# DRAFT: ED Load-Balancing Protocol Memo (for human review)

## Summary
Simulation of {facts['forecast_window']['start']} to {facts['forecast_window']['end']} with
{s['beds']} beds, {s['nurses']} triage nurses and {s['doctors']} doctors. Diversion of
non-urgent patients starts at {s['trigger_pct']}% bed occupancy. Peak forecast demand is
{facts['forecast_window']['peak_forecast_arrivals_per_hour']} arrivals/hour.

## Findings
- Mean wait: {b['mean_wait_min']:.1f} min without diversion vs {w['mean_wait_min']:.1f} min
  with diversion (reduction {_pct(wait_row.get('Result'))}; target
  {_pct(wait_row.get('Target'))}: {wait_row.get('Status', 'n/a')}).
- Capacity-violation rate: {_pct(b['capacity_violation_rate'])} vs
  {_pct(w['capacity_violation_rate'])} (reduction {_pct(viol_row.get('Result'))}; target
  {_pct(viol_row.get('Target'))}: {viol_row.get('Status', 'n/a')}).
- Peak hourly mean wait with diversion: {w['peak_hourly_wait_min']:.1f} min
  (limit {peak_row.get('Target', 45):.0f} min: {peak_row.get('Status', 'n/a')}).
- Peak bed utilization {facts['peak_bed_utilization_pct']}%; {facts['hours_at_diversion_threshold']}
  hours at or above the trigger. {facts['patients_diverted']} patients diverted, up to
  {facts['peak_nonurgent_diversion_pct']}% of non-urgent arrivals in the peak hour.

## Recommended actions
Pre-alert the receiving facilities below before forecast peaks and review the trigger level
with clinical leads.
{fac_lines}

## Limitations
Results come from a simulation on synthetic data and one random seed. High-acuity and
resuscitation cases are excluded from diversion. This is decision support only and does not
replace clinical judgement or hospital protocols. A human must review before any use.
"""


# ------------------------------------------------------------------- number guard
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _collect_numbers(obj, out: list) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(float(m) for m in _NUM.findall(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)


def find_unverified_numbers(text: str, facts: dict) -> list:
    """Numbers in `text` that match no fact (as a value or as a fraction shown as a %)."""
    allowed: list = []
    _collect_numbers(facts, allowed)
    bad = []
    for tok in _NUM.findall(text):
        x = float(tok)
        if x == int(x) and 0 <= x <= 10:      # list numerals, "5 facilities" etc.
            continue
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(x - a) <= tol or abs(x - a * 100) <= tol for a in allowed):
            bad.append(tok)
    return sorted(set(bad))


# ------------------------------------------------------------------------- LLM path
def build_llm_prompt(facts: dict) -> str:
    return "Draft the memo from these verified simulation facts:\n" + json.dumps(
        facts, indent=2, default=str
    )


def generate_memo(
    facts: dict,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: float = LLM_TIMEOUT_S,
    _client=None,
) -> MemoResult:
    template = render_template_memo(facts)
    if _client is None and not api_key:
        return MemoResult(template, "template", "No API key set; used the built-in template.")
    try:
        if _client is None:
            import anthropic  # optional dependency
            _client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=0)
        resp = _client.messages.create(
            model=model,
            max_tokens=900,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_llm_prompt(facts)}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        if not text:
            raise ValueError("empty response")
    except Exception as exc:  # network, auth, timeout, missing package: never break the tab
        return MemoResult(template, "template", f"LLM unavailable ({type(exc).__name__}); used template.")
    return MemoResult(text, "llm", "", find_unverified_numbers(text, facts))