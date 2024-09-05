import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# from smallscout.model import load_model
# from smallscout.preprocessor import preprocess_features


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy Deep Learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!
# This will prove very useful for the Demo Day

#app.state.model = load_model()

# http://127.0.0.1:8000/predict?ticker=AAPL"
@app.get("/predict")
def predict(
        ticker: str,    # "AAPL"
    ):
    """
    Make a single course prediction.
    """

    # 💡 Optional trick instead of writing each column name manually:
    # locals() gets us all of our arguments back as a dictionary
    # https://docs.python.org/3/library/functions.html#locals
    X_pred = pd.DataFrame(locals(), index=[0])

    # model = app.state.model
    # assert model is not None

    # X_processed = preprocess_features(X_pred)
    # y_pred = model.predict(X_processed)

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return dict(prediction=str("yes"))



@app.get("/")
def root():
    return dict(greeting="Hello")
