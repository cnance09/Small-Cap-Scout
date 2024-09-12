import pandas as pd
from fastapi import FastAPI, HTTPException
from smallscout.params import *
from google.cloud import bigquery


#from smallscout.preprocessor import *
from smallscout.preprocessor import preprocess_new_data
from fastapi.middleware.cors import CORSMiddleware
import pickle

# Initialize FastAPI app
app = FastAPI()


# Add middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def get_model_file(model_type, sequence=4, horizon='year-ahead', threshold='50%', small_cap=True):
    if horizon not in ['quarter-ahead', 'year-ahead', 'two-years-ahead']:
        raise ValueError (f"Unsupported horizon: {horizon}, must be quarter-ahead, year-ahead or two-years-ahead")
    if threshold not in ['30%', '50%']:
        raise ValueError (f"Unsupported growth threshold: {threshold}, must be 30% or 50%")

    prep_file = f"models/preprocessor_cross_section.pkl"

    if model_type == 'xgb':
        file_name = f"models/{model_type}_sc{small_cap}_{horizon}_{threshold}.pkl"
    if model_type == 'rnn':
        file_name = f"models/{model_type}_sc{small_cap}_{horizon}_{sequence}_seq_{threshold}.pkl"
    if model_type not in ['xgb', 'rnn']:
        raise ValueError(f"Unknown model type: {model_type}")

    return file_name, prep_file


@app.get("/predict")
def predict(ticker, model_type='xgb', quarter='2024-Q1', sequence=4, horizon='year-ahead', threshold='50%', small_cap=True):
    """
    Predict the worthiness of a stock based on its ticker symbol.
    Fetches data from the local dataset, preprocesses it, and passes it to the logistic regression model.

    Parameters:
        - ticker (str): The stock ticker symbol provided by the user.

    Returns:
        - dict: A JSON object with the prediction result (e.g., worthiness score).
    """

    # Load corresponding model & preprocessor
    model_file, prep_file = get_model_file(model_type=model_type, sequence=sequence, horizon=horizon, threshold=threshold, small_cap=True)
    with open(model_file, 'rb') as f_qrt:
        app.state.model = pickle.load(f_qrt)

    with open(prep_file, 'rb') as f_preprocessor:
        app.state.preprocessor = pickle.load(f_preprocessor)

    # Load data from Google Big Query
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.{BQ_REGION}
        WHERE TICKER = '{ticker}'
        ORDER BY DATE
        """
    client = bigquery.Client(project=GCP_PROJECT)
    query_job = client.query(query)
    result = query_job.result()
    data = result.to_dataframe()
    data = data.astype(DTYPES_RAW)

    if len(data) == 0:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found in dataset.")

    if model_type=='rnn':
        idx = data.index[data.quarter==quarter]
        input_data = data.iloc[(idx-(sequence-1)):(idx)]
    else:
        input_data = data[data.quarter==quarter]
    input_data.drop(columns=['TICKER', 'date', 'quarter', 'name'], inplace=True)

    X_processed = preprocess_new_data(input_data, app.state.preprocessor)

    # Retrieve the preloaded model
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Make the prediction using the logistic regression model
    y_pred = int(model.predict(X_processed)[0])
    y_prob = round(float(model.predict_proba(X_processed)[0][1]),2)

    # Since the logistic regression prediction is likely binary (0 or 1), convert to readable format
    worthiness = "worthy" if y_pred == 1 else "not worthy"

    # Return the prediction result
    results = {"ticker": ticker, "worthiness": worthiness, 'prediction': y_pred, 'probability': y_prob,
            'model_type': model_type, 'quarter':quarter, 'sequence': sequence, 'horizon': horizon, 'threshold': threshold, 'small_cap': small_cap}
    return results


@app.get("/info")
def get_ticker_info(ticker: str):
    """
    Return the latest info (Revenues, market_cap, etc.) for a given stock ticker.
    """

    # Load data from Google Big Query
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.{BQ_REGION}
        WHERE TICKER = '{ticker}'
        ORDER BY DATE DESC
        """
    client = bigquery.Client(project=GCP_PROJECT)
    query_job = client.query(query)
    result = query_job.result()
    data = result.to_dataframe()
    data = data.astype(DTYPES_RAW)

    if data.empty:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found in dataset.")

    latest_data = data.iloc[0, :]

    #ticker_data = app.state.dataset[app.state.dataset['ticker'] == ticker]

    # Ensure the data is sorted by date or some time-based column
    #ticker_data_sorted = ticker_data.sort_values(by='date', ascending=False)

    # Get the latest record
    #latest_data = ticker_data_sorted.iloc[0]

    # Extract the required values
    name = latest_data['name']
    revenues = latest_data['Revenues']  # Adjust column name if needed
    market_cap = latest_data['market_cap']
    OperatingCF = latest_data['NetCashProvidedByUsedInOperatingActivities']
    ProfitLoss = latest_data['ProfitLoss']
    GrossProfit = latest_data['GrossProfit']
    results = {"Ticker": ticker,
        "Company name": name,
        "Market cap": market_cap,
        "Revenues": revenues,
        "Gross Profit": GrossProfit,
        "Net Income": ProfitLoss,
        "Operating Cash Flows": OperatingCF}
    return results


@app.get("/")
def root():
    return {"message": "Welcome to the stock worthiness prediction API!"}


if __name__ == '__main__':
    #print(app.state.dataset.head())
    #print(app.state.model == None)

    print(predict("AAPL", quarter='2023-Q4'))
    # returns --> {'ticker': 'AAPL', 'worthiness': 'not worthy', 'prediction': array([0])}

    print(get_ticker_info('AAPL'))
