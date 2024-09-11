import pandas as pd

def convert_dates(df):
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    return df.drop(columns=['Date'])  # Drop the original 'Date' column after conversion

#print(get_model_file('rnn'))
