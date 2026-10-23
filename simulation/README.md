# Queue simulation

This SimPy model is used by the Streamlit dashboard to compare baseline
operations with a diversion scenario.

## Run

Install the repository-root dependencies and run:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m streamlit run HILOM.py
```

The dashboard compares a no-diversion baseline with an adjustable bed-pressure
diversion scenario using seven days of generated hourly arrivals. This is a
demonstration input, not an SARIMAX forecast.

## Input and outputs

`run_simulation()` accepts an hourly DataFrame with:

- `timestamp`: consecutive hourly timestamps.
- `forecast`: nonnegative expected arrivals in that hour.

It returns a `SimulationResult`:

- `patients`: one row per simulated patient, with stage waits, service times,
  diversion outcome, and time in the system.
- `hourly`: arrivals, outcomes, wait summaries, and sampled resource
  utilization by arrival hour.
- `facilities`: accepted referral counts and simultaneous-capacity limits.

`total_wait` means triage queue wait plus bed queue wait plus provider queue
wait; it excludes service time. The 45-minute target is compared against this
total for each patient.

Example:

```python
from simulation import SimulationConfig, run_simulation

result = run_simulation(
    forecast_df=forecast_df,
    config=SimulationConfig(diversion_occupancy_threshold=0.85),
    facilities=facilities_df,
    seed=42,
)
```

Set `diversion_occupancy_threshold=None` to run the baseline without diversion.
Both scenarios with the same seed use matching arrivals, acuity assignments,
and service-time draws so their results can be compared more fairly.

## Explicit assumptions and limitations

- An hourly forecast is treated as a Poisson expected count; sampled arrivals
  are distributed uniformly within that hour.
- Acuity proportions default to 24% emergent, 47% urgent, and 29% non-urgent.
  Patients are queued FIFO for triage and by acuity priority for beds/providers.
- Triage duration is sampled uniformly from 3–7 minutes. Treatment duration
  follows a lognormal distribution parameterized by a 42-minute arithmetic
  mean and 15-minute standard deviation. These are scenario assumptions, not
  calibrated ED data.
- Staffing is held constant for a run. Although the inflow file includes
  hourly doctor counts, this first version does not yet apply shift changes.
- A bed is held while its patient waits for a provider and receives treatment.
  No additional boarding or discharge delay is modeled because LOS data is not
  present.
- Diversion is offered to non-urgent patients when occupied beds plus the bed
  queue reach the configured threshold. The nearest facility with an available
  simultaneous slot is selected. The slot is held during travel and treatment;
  the ED treatment-time distribution is used as a placeholder at that facility.
- Receiving facilities start empty and are assumed open and willing to accept
  clinically eligible referrals when a slot is available. Their listed
  simultaneous capacities are not hourly referral limits.
- Utilization is estimated from five-minute resource snapshots.

The synthetic dataset contains no observed patient-level waits, treatment
durations, length of stay, initial occupancy, or facility acceptance history.
Accordingly, simulation output is for scenario exploration only and cannot
validate the proposal's waiting-time error targets. Calibrate these assumptions
and routing rules against approved, de-identified operational data before
using the results as operational recommendations.
