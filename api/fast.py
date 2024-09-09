import os
import pandas as pd
from fastapi import FastAPI, HTTPException

from smallcap.registry import load_model
from smallcap.preprocessor import preprocess_features

# Initialize FastAPI app
app = FastAPI()

# Load dataset path from environment variables
QUERY_DATASET = os.environ.get("QUERY_DATASET", "path/data_for_preprocessing.csv")

# Preload the model to avoid reloading it with every request
app.state.model = load_model()

@app.get("/predict")
def predict(ticker: str):
    """
    Predict the worthiness of a stock based on its ticker symbol.
    Fetches data from the local dataset, preprocesses it, and passes it to the logistic regression model.

    Parameters:
        - ticker (str): The stock ticker symbol provided by the user.

    Returns:
        - dict: A JSON object with the prediction result (e.g., worthiness score).
    """
    # Load dataset containing information about all tickers
    try:
        dataset = pd.read_csv(QUERY_DATASET)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")

    # Filter the dataset to get the row for the input ticker
    ticker_data = dataset[dataset['ticker'] == ticker]

    if ticker_data.empty:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found in dataset.")

    # Preprocess the data for model input
    X_processed = preprocess_features(ticker_data)

    # Retrieve the preloaded model
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Make the prediction using the logistic regression model
    y_pred = model.predict(X_processed)

    # Since the logistic regression prediction is likely binary (0 or 1), convert to readable format
    worthiness = "worthy" if y_pred[0] == 1 else "not worthy"

    # Return the prediction result
    return {"ticker": ticker, "worthiness": worthiness}

@app.get("/")
def root():
    return {"message": "Welcome to the stock worthiness prediction API!"}
