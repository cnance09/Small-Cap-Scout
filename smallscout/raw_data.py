## FRED Datasets Loading

import requests
import pandas as pd

# API key and base URL for FRED API
API_KEY = "7bb504adcabc6f374463db2650ad84e5"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
SERIES_IDS = ["BBKMGDP", "FEDFUNDS", "UNRATE", "MEDCPIM158SFRBCLE"]

# Function to fetch data from the FRED API
def fetch_fred_data(series_id):
    params = {
        "series_id": series_id,
        "realtime_start": "2000-01-01",
        "realtime_end": "2024-07-01",
        "api_key": API_KEY,
        "file_type": "json"
    }
    response = requests.get(FRED_URL, params=params)
    if response.status_code == 200:
        return response.json().get('observations', [])
    else:
        print(f"Error fetching {series_id}: {response.status_code}")
        return None

# Fetch and merge data for all series
def load_data():
    date_range = pd.date_range(start='2010-01-01', end='2024-12-31', freq='M')
    merged_df = pd.DataFrame(date_range, columns=['Date']).astype(str)  # Date as string

    for series_id in SERIES_IDS:
        observations = fetch_fred_data(series_id)
        if observations:
            df = pd.DataFrame(observations)[['date', 'value']].rename(columns={'date': 'Date', 'value': f'{series_id}_Value'})
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            merged_df = pd.merge(merged_df, df, on='Date', how='left')

    return merged_df

# Main logic to load and save data
if __name__ == "__main__":
    data = load_data()
    data.to_csv('merged_data.csv', index=False)
    print("Data has been saved to 'merged_data.csv'")
