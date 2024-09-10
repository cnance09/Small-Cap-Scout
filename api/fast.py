import pandas as pd
from fastapi import FastAPI, HTTPException
from smallscout.params import *
from google.cloud import bigquery


#from smallscout.preprocessor import *
from smallscout.preprocessor import preprocess_new_data
from smallscout.utils import get_model_file
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


# Load the models
# Load the pre-trained logistic regression model
#with open(MODEL_PATH, 'rb') as f_qrt:
#    app.state.model = pickle.load(f_qrt)

# Load the preprocessor pipeline
#with open(PREPROCESSOR_PATH, 'rb') as f_preprocessor:
#    app.state.preprocessor = pickle.load(f_preprocessor)

# Load the dataset when the app starts
# Load dataset containing information about all tickers
# try:
#     app.state.dataset = pd.read_csv(QUERY_PATH, index_col=0)
# except Exception as e:
#     raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")


@app.get("/predict")
def predict(ticker, model_type='logistic_regression', quarter='2024-Q2', sequence=4, horizon='year-ahead', threshold='50%', small_cap=True):
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
    try:
        client = bigquery.Client(project=GCP_PROJECT)
        query_job = client.query(query)
        result = query_job.result()
        data = result.to_dataframe()
    except:
        if (len(data) == 0)|(data == None):
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found in dataset.")
        raise ValueError('Error Loading Dataset')
    print(model_file, prep_file)


    if model_type=='RNN':
        idx = data.index[data.quarter==quarter]
        input_data = data.iloc[(idx-3):(idx)]
    else:
        input_data = data[data.quarter==quarter]
    input_data.drop(columns=['TICKER', 'date', 'quarter'], inplace=True)


    # Filter the dataset to get the row for the input ticker
    # ticker_data =  app.state.dataset[app.state.dataset['TICKER'] == ticker]
    # ticker_data['target'] = 0
    # if ticker_data.empty:
    #     raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found in dataset.")

    # Preprocess the data for model input

    # Preprocess the data using the loaded preprocessor
    # preprocessor = app.state.preprocessor
    # X_processed = preprocess_new_data(ticker_data, preprocessor)

    X_processed = preprocess_new_data(input_data, app.state.preprocessor)

    # Retrieve the preloaded model
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Make the prediction using the logistic regression model
    y_pred = model.predict(X_processed)

    # Since the logistic regression prediction is likely binary (0 or 1), convert to readable format
    worthiness = "worthy" if y_pred[0] == 1 else "not worthy"

    # Return the prediction result
    return {"ticker": ticker, "worthiness": worthiness, 'prediction': y_pred}

@app.get("/")
def root():
    return {"message": "Welcome to the stock worthiness prediction API!"}

#print(app.state.dataset.head())

#print(app.state.model == None)

print(predict("AAPL", quarter='2023-Q4'))
# returns --> {'ticker': 'AAPL', 'worthiness': 'not worthy', 'prediction': array([0])}
