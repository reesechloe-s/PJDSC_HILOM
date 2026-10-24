"""Policy-memo generation for the dashboard's 'Policy Memo' tab.

Design rules (from the proposal: LLM drafts from *verified* model outputs, human reviews):
  1. Every number comes from `build_memo_facts`, i.e. from the simulation, never from the LLM.
  2. A deterministic template memo always works (no key, no network, instant).
  3. If a Google AI Studio (Gemini) API key is set, the LLM rewrites the same facts; any
     number it uses that is not in the facts is flagged for the reviewer.
No Streamlit imports here, so the module is unit-testable.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from groq import Groq

# Update these lines near the top of your memo file:
DEFAULT_MODEL = os.environ.get("GROQ_LLM_MODEL", "qwen/qwen3.8-27b")
FALLBACK_MODEL = os.environ.get("GROQ_LLM_FALLBACK_MODEL", "qwen/qwen3.8-27b")
RETRY_DELAY_S = 1.0
API_KEY_NAMES = ("GROQ_API_KEY",)
LLM_TIMEOUT_S = 30.0

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


def read_env_key(path, name: str = "GROQ_API_KEY") -> str | None:
    """Read NAME from a .env file without extra dependencies. Streamlit does not load .env itself.
    Handles a Windows BOM, `export NAME=...`, quotes and trailing comments."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, sep, val = line.partition("=")
        if sep and key.strip() == name:
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            else:
                val = val.split(" #")[0].strip()
            return val or None
    return None


def get_api_key(env_paths=(), secrets=None) -> str | None:
    """First key found in: real environment, each .env path, then a secrets mapping."""
    for name in API_KEY_NAMES:
        if os.environ.get(name):
            return os.environ[name]
    for path in env_paths:
        for name in API_KEY_NAMES:
            key = read_env_key(path, name)
            if key:
                return key
    if secrets is not None:
        for name in API_KEY_NAMES:
            try:
                key = secrets.get(name)
            except Exception:
                key = None
            if key:
                return key
    return None


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
    fac_cols = [c for c in ("facility_name", "travel_time_mins", "max_simultaneous_patients",
                            "simultaneous_capacity", "referrals_accepted")
                if c in facility_results.columns]
    facilities = (
        facility_results[fac_cols].sort_values("referrals_accepted", ascending=False)
        if "referrals_accepted" in fac_cols else facility_results[fac_cols]
    ).head(5)
    facts = {
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
    stage_cols = {"triage": "mean_triage_wait", "bed": "mean_bed_wait", "doctor": "mean_provider_wait"}
    stages = {k: round(float(sim_df[c].fillna(0).mean()), 1)
              for k, c in stage_cols.items() if c in sim_df.columns}
    if stages:  # average of the hourly means, not patient-weighted
        facts["avg_hourly_wait_by_stage_min"] = stages
    for key, col in (("peak_doctor_utilization_pct", "mean_provider_utilization"),
                     ("peak_nurse_utilization_pct", "mean_nurse_utilization")):
        if col in sim_df.columns:
            facts[key] = round(float(sim_df[col].fillna(0).max()) * 100, 1)
    return facts


# ------------------------------------------------------------------- template memo
def _pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def render_template_memo(facts: dict) -> str:
    s, b, w = facts["settings"], facts["baseline"], facts["with_diversion"]
    t = {row["KPI"]: row for row in facts["targets"]}
    wait_row = t.get("Mean wait reduction vs baseline", {})
    viol_row = t.get("Capacity-violation reduction vs baseline", {})
    peak_row = next((r for k, r in t.items() if k.startswith("Peak hourly")), {})
    triggered = facts["patients_diverted"] > 0
    fw = facts["forecast_window"]
    stages = facts.get("avg_hourly_wait_by_stage_min")
    stage_line, dominant_note = "", ""
    if stages:
        stage_line = (
            f"\n- Average hourly mean wait by stage: triage {stages.get('triage', 'n/a')} min, "
            f"bed {stages.get('bed', 'n/a')} min, doctor {stages.get('doctor', 'n/a')} min. "
            f"Peak doctor utilization {facts.get('peak_doctor_utilization_pct', 'n/a')}%, "
            f"peak triage-nurse utilization {facts.get('peak_nurse_utilization_pct', 'n/a')}%."
        )
        top = max(stages, key=stages.get)
        if stages[top] > 0:
            dominant_note = f" The longest average wait is at the {top} stage."

    if triggered:
        fac_lines = "\n".join(
            f"- {f.get('facility_name', '?')}: {f.get('referrals_accepted', 0)} referrals "
            f"accepted, {f.get('travel_time_mins', '?')} min travel"
            for f in facts["top_receiving_facilities"]
        ) or "- No facility data available."
        actions = (
            "Pre-alert the receiving facilities below before forecast peaks and review the "
            "trigger level with clinical leads.\n" + fac_lines
        )
        diversion_note = ""
    else:
        actions = (
            f"Diversion was not triggered under these settings: peak bed utilization was "
            f"{facts['peak_bed_utilization_pct']}% against a {s['trigger_pct']}% trigger. "
            "Baseline and diversion results are therefore identical, and this simulation does "
            "not show whether diversion helps. Any waits above target come from something "
            "other than bed saturation." + dominant_note + " Review doctor and triage staffing and consider "
            "a trigger based on queue length or wait time instead of bed occupancy."
        )
        diversion_note = " Diversion never activated, so both scenarios are the same run."

    return f"""# DRAFT: ED Load-Balancing Protocol Memo (for human review)

## Summary
Simulation of {fw['start']} to {fw['end']} with {s['beds']} beds, {s['nurses']} triage nurses and {s['doctors']} doctors. Diversion of non-urgent patients starts at {s['trigger_pct']}% bed occupancy. Peak forecast demand is {fw['peak_forecast_arrivals_per_hour']} arrivals/hour.

## Findings
- Mean wait: {b['mean_wait_min']:.1f} min without diversion vs {w['mean_wait_min']:.1f} min with diversion. Reduction {_pct(wait_row.get('Result'))}, target {_pct(wait_row.get('Target'))}: {wait_row.get('Status', 'n/a')}.
- Capacity-violation rate: {_pct(b['capacity_violation_rate'])} vs {_pct(w['capacity_violation_rate'])}. Reduction {_pct(viol_row.get('Result'))}, target {_pct(viol_row.get('Target'))}: {viol_row.get('Status', 'n/a')}.
- Peak hourly mean wait with diversion: {w['peak_hourly_wait_min']:.1f} min (limit {peak_row.get('Target', 45):.0f} min: {peak_row.get('Status', 'n/a')}).
- Peak bed utilization {facts['peak_bed_utilization_pct']}%; {facts['hours_at_diversion_threshold']} hours at or above the trigger; {facts['patients_diverted']} patients diverted (peak hour: {facts['peak_nonurgent_diversion_pct']}% of non-urgent arrivals).{diversion_note}{stage_line}

## Recommended actions
{actions}

## Limitations
Results come from a simulation on synthetic data and one random seed. High-acuity and resuscitation cases are excluded from diversion. This is decision support only and does not replace clinical judgement or hospital protocols. A human must review before any use.
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


def _is_transient(exc: Exception) -> bool:
    """API overload/rate limit: worth one retry on the same model."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    err_name = type(exc).__name__
    return status in (429, 500, 502, 503, 504) or err_name in (
        "ServerError", "RateLimitError", "InternalServerError", "APIStatusError", "APIConnectionError"
    )


def _describe(exc: Exception, api_key: str | None) -> str:
    msg = " ".join(str(exc).split())[:220]
    if api_key:
        msg = msg.replace(api_key, "***")
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def generate_memo(
    facts: dict,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: float = LLM_TIMEOUT_S,
    _client=None,
) -> MemoResult:
    """Groq LLM memo if possible, else the template. Per model: one retry on 5xx/429. If the
    primary model fails, the fallback model is tried. Never raises."""
    template = render_template_memo(facts)
    if _client is None and not api_key:
        return MemoResult(template, "template", "No API key set; used the built-in template.")

    try:
        if _client is None:
            _client = Groq(api_key=api_key, timeout=timeout)
    except Exception as exc:
        return MemoResult(template, "template", f"LLM unavailable ({_describe(exc, api_key)}); used template.")

    prompt = build_llm_prompt(facts)
    models = [model] + ([FALLBACK_MODEL] if FALLBACK_MODEL and FALLBACK_MODEL != model else [])
    last: Exception | None = None
    
    for name in models:
        for attempt in range(2):
            try:
                chat_completion = _client.chat.completions.create(
                    model=name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_completion_tokens=4096,
                )
                text = (chat_completion.choices[0].message.content or "").strip()
                if not text:
                    raise ValueError("empty or blocked response")
                note = "" if name == model else f"Primary model {model} failed; used {name}."
                return MemoResult(text, "llm", note, find_unverified_numbers(text, facts))
            except Exception as exc:
                last = exc
                if attempt == 0 and _is_transient(exc):
                    time.sleep(RETRY_DELAY_S)
                    continue
                break
                
    return MemoResult(template, "template", f"LLM unavailable ({_describe(last, api_key)}); used template.")