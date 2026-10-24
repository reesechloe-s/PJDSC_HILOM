# HILOM: Hospital Inflow and Optimization Model

## Problem Statement
Public hospital Emergency Departments (EDs)—such as the Philippine General Hospital (UP-PGH)—frequently operate at or beyond capacity due to high volumes of unscheduled walk-ins. This saturation leads to extended wait times, resource exhaustion, and delayed care, particularly for predominantly indigent patient populations. Furthermore, external environmental shocks, such as severe weather events, can unpredictably alter healthcare-seeking behavior, causing sudden post-rain arrival surges that overwhelm fixed operational capacities.

## Project Description
HILOM is a decision-support simulation tool designed to test operational interventions and prevent ED saturation. In the absence of live electronic health record (EHR) telemetry, HILOM utilizes a synthetic data pipeline built on historical census metrics and meteorological data to forecast patient arrivals.

At its core, HILOM features a Discrete-Event Simulation (DES) engine that models patient triage and treatment queues. By applying an automated, capacity-driven diversion protocol, the system routes non-urgent patients to alternative local government unit (LGU) facilities before the primary hospital reaches critical saturation.

## Key Features

**Synthetic Data Generation**
* Generates baseline Poisson arrivals based on 2019 UP-PGH metrics (~8.4 patients/hr).
* Applies an ETCCDI-based "wet hour" threshold (98th percentile) to model 2-hour post-rain arrival rebounds.
* Adjusts volumes dynamically using day-of-week and time-of-day multipliers.

**Forecasting**
* Utilizes a SARIMAX (1,0,0)(1,0,0,24) time-series model to predict expected hourly arrivals 24 hours to 7 days ahead.
* Incorporates multiple predictors including hour of the day, day of the week, precipitation, heavy-rain flags, and delayed post-rain effects.
* Generates expected hourly arrivals and nominal 95% prediction intervals (achieving an MAE of 2.35), which serve as direct inputs for the discrete-event queue simulation.

**Discrete-Event Simulation (DES) Engine**
* Simulates priority-based queuing (Emergent > Urgent > Non-urgent) using `SimPy`.
* Configurable scenario testing for staffing levels (e.g., 70 beds, 13 nurses, 4 doctors) and statistical service distributions.
* Granular wait-time tracking against the target wait time.

**Network Load Balancing & Triage Routing**
* Monitors real-time internal bed pressure and care provider capacity.
* Automatically triggers network diversion when bed occupancy or provider threshold reaches configurable thresholds (e.g., 85%).
* Routes non-urgent patients to the nearest available LGU facility optimized for the shortest travel time and active receiving capacity.

**Automated Memo Deployment**
* Integrates the Groq LLM to instantly translate optimization metrics and capacity breaches into actionable triage memos for staff review and deployment.

**Interactive Dashboard**
* Built with Streamlit to visualize queuing bottlenecks, resource utilization, and diversion impact in real-time.
* Allows users to dynamically configure scenario parameters directly from the UI, including hospital capacity, staffing levels, and diversion trigger thresholds.

---

## Instructions for Use (Streamlit)

### 1. Prerequisites
Ensure you have Python 3.10+ installed. It is recommended to use a virtual environment.
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Dependencies
Install the required packages, including `simpy`, `streamlit`, `pandas`, and `numpy`.
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
Launch the Streamlit interface to configure your scenario parameters and run the simulation.
```bash
streamlit run app.py
```
*Note: A browser window will open automatically at localhost. Once open, you can use the sidebar to manually configure capacity constraints, resource levels, and diversion thresholds before running the simulation.*
