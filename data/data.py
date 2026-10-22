# NOTES:
# Use poisson distri to simulate random, unscheduled walk-ins (that's why ER, since OPD usually has scheduled appointments), different queuing mathematics
# Optimization engine is designed to prevent emergency dept from reaching saturation (to help, especially dun sa Inquirer article)
## - So constraints - 70 ER beds, 13 nurses per shift, 3 to 4 doctors per shift
## - Sourced from:
##   1. "Determination of emergency department patient utilization and staffing at the University of the Philippines-Philippine General Hospital (UP-PGH)" (Tabuñar & Pagkatipunan, 2023)
##      - 13.2 nurses per 8-hour shift, 4 morning / 3 night doctors on 12-hour shifts
##      - Day-of-week trends (Monday/Friday spikes, Wednesday/weekend drops)
##      - Acuity breakdown: 24% emergent, 47% urgent, 29% non-urgent
##      - Shift
##   2. "Facts & Figures 2020" (Philippine General Hospital, 2020)
##      - Total pre-pandemic 2019 ER census = 73,529 emergency patients (yielding ~8.4 patients/hour baseline)
##      - Class D indigent profile (88.48%) justifying public LGU receiving facilities
# Source for the rain stuff
## Alkhalidi, M., Al-Dabbous, A., Al-Dabbous, S., & Alzaid, D. (2025). Evaluating the accuracy of the ERA5 model in predicting wind speeds across coastal and offshore regions. Journal of Marine Science and Engineering, 13(1), 149. https://doi.org/10.3390/jmse13010149

# On pediatric hourly data, due to limited pediatric hourly data, pediatric arrival variance is assumed to mirror the adult temporal distributions identified by Tabuñar & Pagkatipunan (2023)

import numpy as np
import pandas as pd
import requests
import time

def generate_synthetic_hospital_data(start_date='2021-01-01', end_date='2023-12-31'):
    # 1. Hourly Time Series
    time_index = pd.date_range(start=start_date, end=pd.Timestamp(end_date) + pd.Timedelta(days=1), freq='h', inclusive='left')
    df = pd.DataFrame({'timestamp': time_index})
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Mon, 6=Sun
    df['month'] = df['timestamp'].dt.month

    # 2. Seasonality & Exogenous Variables
    # 8.4 patients/hour based on 73,529 total emergency patients in 2019 (PGH Facts & Figures 2020)
    base_lambda = 8.39
    
    # Initialized the local random number generator
    rng = np.random.default_rng(42)

    # One multiplier per calendar date to eliminate hourly noise
    unique_dates = df[['date', 'day_of_week']].drop_duplicates()
    def assign_daily_multiplier(dow):
        if dow in [0, 4]: # If monday (1.05), fri (1.063)
            return rng.uniform(1.04, 1.07)
        elif dow in [2, 5, 6]: # if wed (0.954), sat (0.96), sun (abt 0.95)
            return rng.uniform(0.94, 0.97)
        else: 
            return rng.uniform(1.00, 1.03) # tues abt 1.025,thurs abt 1.007
    
    unique_dates['dow_multiplier'] = unique_dates['day_of_week'].apply(assign_daily_multiplier)
    df = df.merge(unique_dates[['date', 'dow_multiplier']], on='date', how='left')

    # Baseline is derived from the 2019 annual emergency census, with daily and hourly multipliers added to simulate variation.
    def get_tod_multiplier(hour):
        if 6 <= hour < 14: # from 6am to 2pm 
            return 1.1
        elif 14 <= hour < 22: # 2pm to 10pm, (the 8 hour shift census revealed that this has highest number of consultations, Tabunar paper.)
            return 1.15
        else: # night shift 10pm to 6am
            return 0.75

    df['tod_multiplier'] = df['hour'].apply(get_tod_multiplier)

    # 3. Weather Events (Real Manila Historical Data via Open-Meteo)
    print("Fetching real historical weather data for Manila in yearly chunks...")
    
    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year
    weather_dfs = []
    
    for year in range(start_year, end_year + 1):
        print(f"  -> Requesting data for {year}...")
        
        chunk_start = f"{year}-01-01"
        chunk_end = f"{year}-12-31"
        
        if year == end_year:
            chunk_end = end_date
            
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude=14.58&longitude=120.98&"
            f"start_date={chunk_start}&end_date={chunk_end}&"
            f"hourly=precipitation&timezone=Asia%2FSingapore"
        )
        
        success = False
        # Try up to 3 times per year if the API drops the connection
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                if 'hourly' not in data:
                    raise ValueError(f"Missing 'hourly' field in API response for {year}.")
                
                temp_df = pd.DataFrame({
                    'timestamp': pd.to_datetime(data['hourly']['time']),
                    'precipitation_mm': data['hourly']['precipitation']
                })
                weather_dfs.append(temp_df)
                success = True
                
                # Sleep for 2 seconds to avoid triggering rate limits
                time.sleep(2) 
                break # Break out of the retry loop if successful
                
            except Exception as e:
                print(f"    [Attempt {attempt + 1}/3] failed: {e}")
                time.sleep(3) # Wait 3 seconds before retrying
                
        # If it fails all 3 attempts, fill this specific year with 0mm fallback data
        if not success:
            print(f"    WARNING: API completely failed for {year}. Using 0mm fallback.")
            chunk_index = pd.date_range(start=chunk_start, end=pd.Timestamp(chunk_end) + pd.Timedelta(days=1), freq='h', inclusive='left')
            temp_df = pd.DataFrame({
                'timestamp': chunk_index,
                'precipitation_mm': np.zeros(len(chunk_index))
            })
            weather_dfs.append(temp_df)
            
    weather_df = pd.concat(weather_dfs, ignore_index=True)
    weather_df = weather_df.drop_duplicates(subset=['timestamp'])
    print("All weather data processing complete!")

    # --- WEATHER DATA CHECK ---
    print("\n--- WEATHER DATA CHECK ---")
    print(f"Target row count (Main DF): {len(df)} hours")
    print(f"Fetched row count (Weather DF): {len(weather_df)} hours")
    
    # Check if there are any missing values in the fetched precipitation column
    missing_rain = weather_df['precipitation_mm'].isna().sum()
    print(f"Missing API values: {missing_rain}")
    # ------------------------------------
    
    # Merge the real weather data into our main hourly dataframe
    df = pd.merge(df, weather_df, on='timestamp', how='left')
    
    # Fill any missing API data with 0 (no rain)
    df['precipitation_mm'] = df['precipitation_mm'].fillna(0)
    
    # -------------------------------------------------------------------------
    # Synthetic scenario assumption:
    # Extreme rainfall may delay healthcare-seeking behaviour,
    # leading to short-term arrival rebounds after rainfall subsides.
    # JUSTIFICATION: TWO-FEATURE WEATHER APPROACH
    # 1. Continuous (precipitation_mm): Retained so the SARIMAX model can learn 
    #    linear, baseline shifts in patient behavior during regular rainy days.
    # 2. Binary Shock (is_heavy_rain): Triggers when a severe anomaly occurs
    #
    # To bypass the ERA5 satellite's 25km grid-smoothing limitation, we apply the 
    # ETCCDI statistical methodology: isolating "wet hours" (>= 1.0 mm).
    # the 98th percentile (assumption, correction method for this project only). 
    # -------------------------------------------------------------------------
    
    # Isolate "wet hours" according to ETCCDI standards
    wet_hours = df[df['precipitation_mm'] >= 1.0]['precipitation_mm']
    if len(wet_hours) > 0:
        heavy_rain_threshold = wet_hours.quantile(0.98)
    else:
        heavy_rain_threshold = float("inf")
    
    # Engineer the binary shock feature
    df['is_heavy_rain'] = (df['precipitation_mm'] >= heavy_rain_threshold).astype(int)
    
    if np.isfinite(heavy_rain_threshold):
        print(
            f"ETCCDI 'Wet Hour' 98th Percentile Threshold set at: "
            f"{heavy_rain_threshold:.2f} mm"
        )
    else:
        print("No wet hours detected. Heavy rain threshold disabled.")
    
    # Vectorized post-rain surge: flags the 2 hours immediately following heavy rain
    # This is much faster and cleaner than a while loop
    df['post_rain_surge'] = (
        (df['is_heavy_rain'].shift(1).fillna(0) == 1) | 
        (df['is_heavy_rain'].shift(2).fillna(0) == 1)
    ).astype(int)

    # Modify the base lambda calculation to include the real weather
    df['lambda'] = base_lambda * df['dow_multiplier'] * df['tod_multiplier']
    
    # Base arrivals + delayed rain surge
    # TODO IN PROCESSING: Poisson distribution is kept as the baseline for this aggregated hourly dataset. 
    # Over-dispersion (clustering from mass casualties, holidays) will be processed later during stress-testing scenarios.
    df['total_arrivals'] = rng.poisson(df['lambda'])
    surge_mask = df['post_rain_surge'] == 1
    df.loc[surge_mask, 'total_arrivals'] += rng.poisson(4, sum(surge_mask))

    # 4. Patient Categories 
    # Emergency, Urgent, Non-urgent based on Tabunar's paper.
    category_probs = [0.24, 0.47, 0.29]
    
    cat_matrix = np.array([
        rng.multinomial(n, category_probs) if n > 0 else np.zeros(3, dtype=int)
        for n in df['total_arrivals']
    ])

    df['acuity_emergent'] = cat_matrix[:, 0]    # Excluded from diversion
    df['acuity_urgent'] = cat_matrix[:, 1]      # Standard ER queue
    df['acuity_non_urgent'] = cat_matrix[:, 2]  # Primary targets for diversion

    df['total_er_beds'] = 70
    df['nurses_on_duty'] = 13
    df['doctors_on_duty'] = np.where(df['hour'].between(7, 18), 4, 3)

    # REMOVED: Static wait times, service durations, and load_factors.
    # TODO IN PROCESSING: Patient length of stay, real bed occupancy, admissions, discharges, and individual 
    # service durations will be processed later on in the Discrete Event Simulation (DES) pipeline step. 
    # The optimization engine will calculate wait times dynamically from the actual queue.

    # Cleanup (precipitation_mm and is_heavy_rain remain in the dataframe)
    df = df.drop(columns=['dow_multiplier', 'tod_multiplier', 'lambda', 'date', 'post_rain_surge'])
    return df

def generate_alternative_facilities():
    # Public health network facilities selected due to 88.5% Class D indigent profile (PGH Facts & Figures 2020).
    # So we chose public hospitals and health centers with no fees.
    # TODO: Diversion policy tracking (eligible patients, rejected referrals, residual demand) will be processed later in the optimization engine step.
    data = {
        'facility_id': ['ALT_1', 'ALT_2', 'ALT_3', 'ALT_4', 'ALT_5'],
        'facility_name': [
            'Pedro Gil Health Center', 
            'Paco Health Center',
            'San Andres Health Center',
            'Ospital ng Maynila Medical Center', 
            'Santa Ana Hospital'
        ],
        'facility_type': [
            'LGU Health Center', 
            'LGU Health Center', 
            'LGU Health Center', 
            'City Hospital', 
            'City Hospital'
        ],
        'address': [
            '1423 A. Francisco St, San Andres Bukid, Manila, 1017 Metro Manila',
            '1427, Canonigo Street, Paco, Manila, 1007 Metro Manila',
            '1313 San Andres St, Malate, Manila, 1004 Kalakhang Maynila',
            '719 Quirino Avenue, corner A. Mabini St, Malate, Manila, 1004 Metro Manila',
            'New Panaderos St, Santa Ana'
        ],
        # Approximate road distances from UP-PGH Emergency Department
        'distance_km': [2.5, 2.8, 2.3, 2.4, 3.8],
        'travel_time_mins': [14, 9, 12, 10, 19],

        # Larger city hospitals have higher simultaneous capacities than LGU health centers
        # these are just baselines to train model. it can be configured in the streamlit dashboard by the user later on, to offer dynamic decision support
        'max_simultaneous_patients': [4, 5, 6, 15, 10],

        # Coordinates for Streamlit dashboard mapping
        'lat': [14.572321281906392, 14.582308150548323, 14.573456427860924, 14.564653497214051, 14.583615345654698],
        'lon': [121.00040706911491, 120.99770911118813, 120.99874964417857, 120.98701092883572, 121.01642389814997]
        
    }
    return pd.DataFrame(data)

synthetic_data = generate_synthetic_hospital_data()
alternative_facilities = generate_alternative_facilities()

synthetic_data.to_parquet('synthetic_hospital_inflow.parquet', index=False)
alternative_facilities.to_parquet('alternative_facilities.parquet', index=False)


## == CHECKS TO KNOW IF SCRIPT FAILS == 
# Expected number of hourly records: 3 complete calendar years
expected_hours = len(
    pd.date_range(
        start='2021-01-01',
        end='2023-12-31 23:00:00',
        freq='h'
    )
)

assert len(synthetic_data) == expected_hours # verify if no hours were skipped/dropped
assert synthetic_data['timestamp'].is_unique # no duplicate timestamps
assert synthetic_data['timestamp'].is_monotonic_increasing 

# Acuity counts must sum to total arrivals in every hour
acuity_total = (
    synthetic_data['acuity_emergent']
    + synthetic_data['acuity_urgent']
    + synthetic_data['acuity_non_urgent']
)

assert acuity_total.equals(synthetic_data['total_arrivals'])

# Counts must be nonnegative
count_columns = [
    'total_arrivals',
    'acuity_emergent',
    'acuity_urgent',
    'acuity_non_urgent'
]

assert (synthetic_data[count_columns] >= 0).all().all()