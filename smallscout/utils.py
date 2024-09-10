import pandas as pd

def convert_dates(df):
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    return df.drop(columns=['Date'])  # Drop the original 'Date' column after conversion

def get_model_file(model_type, sequence=4, horizon='year-ahead', threshold='50%', small_cap=True):
    if horizon not in ['quarter-ahead', 'year-ahead', 'two-years-ahead']:
        raise ValueError (f"Unsupported horizon: {horizon}, must be quarter-ahead, year-ahead or two-years-ahead")
    if threshold not in ['30%', '50%']:
        raise ValueError (f"Unsupported growth threshold: {threshold}, must be 30% or 50%")

    if model_type in ['logistic_regression', 'knn', 'svc', 'mlpclassifier']:
        file_name = f"Models/{model_type}_sc{small_cap}_{horizon}_{threshold}.pkl"
        prep_file = f"Models/preprocessor_cross_section.pkl"
        return file_name, prep_file
    if model_type == 'rnn':
        file_name = f"Models/{model_type}_sc{small_cap}_{horizon}_{sequence}_seq_{threshold}.pkl"
        prep_file = f"Models/prepocessor_rnn.pkl"
        return file_name, prep_file

    raise ValueError(f"Unknown model type: {model_type}")

#print(get_model_file('rnn'))
