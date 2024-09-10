import argparse
import requests
import pandas as pd
from datetime import datetime
import yfinance as yf
import time
from tqdm import tqdm
import os

## FRED Datasets Loading
# API key and base URL for FRED API
API_KEY = "7bb504adcabc6f374463db2650ad84e5"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
SERIES_IDS = ["BBKMGDP", "FEDFUNDS", "UNRATE", "MEDCPIM158SFRBCLE"]

## FRED Datasets Loading

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

# Function to load and merge FRED data
def load_fred_data():
    date_range = pd.date_range(start='2010-01-01', end='2024-12-31', freq='M')
    merged_df = pd.DataFrame(date_range, columns=['Date']).astype(str)

    for series_id in SERIES_IDS:
        observations = fetch_fred_data(series_id)
        if observations:
            df = pd.DataFrame(observations)[['date', 'value']].rename(columns={'date': 'Date', 'value': f'{series_id}_Value'})
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            merged_df = pd.merge(merged_df, df, on='Date', how='left')

    return merged_df

## Yahoo dataset

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start="2010-01-01", end=datetime.now().strftime("%Y-%m-%d"))
        data = data.reset_index()
        data['Ticker'] = ticker
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

# Batch processing function for stock data
def process_batch(tickers, batch_size=1800, time_limit=3600):
    results = []
    start_time = time.time()

    for i, ticker in enumerate(tqdm(tickers)):
        results.append(fetch_stock_data(ticker))

        if (i + 1) % batch_size == 0:
            elapsed_time = time.time() - start_time
            if elapsed_time < time_limit:
                time.sleep(time_limit - elapsed_time)
            start_time = time.time()

    return pd.concat(results, ignore_index=True)

# Function to process stock data in batches
def process_stock_data():
    # Read the CSV file containing tickers
    df = pd.read_csv('~/Small-Cap-Scout/raw_data/cik_ticker_pairs.csv')

    # Get the list of tickers
    tickers = df['TICKER'].tolist()

    all_data = pd.DataFrame()
    batch_size = 1800  # Slightly under 2000 to account for potential errors

    # Process tickers in batches
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} of {len(tickers)//batch_size + 1}")
        batch_data = process_batch(batch)
        all_data = pd.concat([all_data, batch_data], ignore_index=True)

        # Save intermediate results
        all_data.to_csv(f'yahoo_stock_data_since_2010_batch_{i//batch_size + 1}.csv', index=False)

    # Save final results
    all_data.to_csv('yahoo_stock_data_since_2010_complete.csv', index=False)
    print("Data collection complete. Final results saved to 'yahoo_stock_data_since_2010_complete.csv'")

# Main function with command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Select the data loading task")
    parser.add_argument(
        "--task",
        choices=["fred", "yahoo"],
        required=True,
        help="Specify which task to run: 'fred' for FRED data, 'yahoo' for Yahoo stock data",
    )

    args = parser.parse_args()

    if args.task == "fred":
        print("Loading FRED data...")
        fred_data = load_fred_data()
        fred_data.to_csv('merged_data.csv', index=False)
        print("FRED data has been saved to 'merged_data.csv'")
    elif args.task == "yahoo":
        print("Loading Yahoo Finance stock data...")
        process_stock_data()

if __name__ == "__main__":
    main()
