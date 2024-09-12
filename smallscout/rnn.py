# Import Packages

import pandas as pd
import numpy as np
import datetime
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from smallscout.preprocessor import preprocess_new_data

def train_test_split_rnn(df):
    # Train_Test Split
    unique_groups = df['TICKER'].unique()
    train_groups, test_groups = train_test_split(unique_groups, test_size=0.3, random_state=42)

    data_train = df[df['TICKER'].isin(train_groups)]
    data_test = df[df['TICKER'].isin(test_groups)]

    return data_train, data_test, train_groups, test_groups

def preprocess_training_data(X_train, preprocessor):
    """Fits and transforms the training data using the provided pipeline."""
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    return X_train_processed, preprocessor

def preproc_rnn(X_train_pp, X_test_pp=None, quarters_input=4, horizon='quarter-ahead'):

    if horizon == 'quarter-ahead':
        adjustment = 0
    elif horizon == 'year-ahead':
        adjustment = 3
    elif horizon == 'two-years-ahead':
        adjustment = 7

    if X_test_pp == None:
        X_train_sequences = []
        y_train_sequences = []


        for company in X_train_pp.groupby(['remainder__cik', 'remainder__TICKER']):
            for i, _ in enumerate(company[1].iterrows()):
                if i+quarters_input+1+adjustment > len(company[1]):
                    break
                sequence = company[1].iloc[i:i+quarters_input, :-4]
                target = company[1].iloc[i+quarters_input+adjustment, -1]
                X_train_sequences.append(sequence)
                y_train_sequences.append(target)
        X_train_sequences = np.array(X_train_sequences).astype('float32')
        y_train_sequences = np.array(y_train_sequences).astype('float32')

        return X_train_sequences, y_train_sequences

    X_train_sequences = []
    y_train_sequences = []
    X_test_sequences = []
    y_test_sequences = []

    for company in X_train_pp.groupby(['remainder__cik', 'remainder__TICKER']):
        for i, _ in enumerate(company[1].iterrows()):
            if i+quarters_input+1+adjustment > len(company[1]):
                break
            sequence = company[1].iloc[i:i+quarters_input, :-4]
            target = company[1].iloc[i+quarters_input+adjustment, -1]
            X_train_sequences.append(sequence)
            y_train_sequences.append(target)
    X_train_sequences = np.array(X_train_sequences).astype('float32')
    y_train_sequences = np.array(y_train_sequences).astype('float32')

    for company in X_test_pp.groupby(['remainder__cik', 'remainder__TICKER']):
        for i, _ in enumerate(company[1].iterrows()):
            if i+quarters_input+1+adjustment > len(company[1]):
                break
            sequence = company[1].iloc[i:i+quarters_input, :-4]
            target = company[1].iloc[i+quarters_input+adjustment, -1]
            X_test_sequences.append(sequence)
            y_test_sequences.append(target)
    X_test_sequences = np.array(X_test_sequences).astype('float32')
    y_test_sequences = np.array(y_test_sequences).astype('float32')

    return X_train_sequences, y_train_sequences, X_test_sequences, y_test_sequences

def run_RNN(df, quarters_input=4, threshold=0.5, small_cap=True, horizon='quarter'):
    # Set model according to given parameters
    if horizon == 'quarter-ahead':
        col = 'mc_qtr_growth_pct'
        adjustment = 0
    elif horizon == 'year-ahead':
        col = 'mc_yr_growth_pct'
        adjustment = 3
    elif horizon == 'two-years-ahead':
        col = 'mc_2yr_growth_pct'
        adjustment = 7

    final_activation = 'sigmoid'
    metrics=['accuracy', 'precision', 'recall']

    if small_cap==True:
        target_func = lambda x: 1 if ((x[col] > threshold) & (x.small_cap == 1)) else 0
    else:
        target_func = lambda x: 1 if ((x[col] > threshold)) else 0

    df['target'] = df.apply(target_func, axis=1)

    # Train_Test Split
    data_train, data_test, train_groups, test_groups = train_test_split_rnn(df)

    # Remove columns
    cols_drop = ['CIK',
                 'mc_qtr_growth',
                 'mc_qtr_growth_pct',
                 'mc_yr_growth',
                 'mc_yr_growth_pct',
                 'mc_2yr_growth',
                 'mc_2yr_growth_pct',
                 'date',
                 'year']

    X_train = data_train[data_train['TICKER'].isin(train_groups)].drop(columns=cols_drop).reset_index(drop=True)
    X_test = data_test[data_test['TICKER'].isin(test_groups)].drop(columns=cols_drop).reset_index(drop=True)

    # Preprocess X_train and X_test
    with open('models/preprocessor_cross_section.pkl', 'rb') as f_preprocessor:
        preprocessor = pickle.load(f_preprocessor)

    X_train_pp, preprocessor = preprocess_training_data(X_train, preprocessor=preprocessor)
    X_train_pp = pd.DataFrame(X_train_pp, columns=preprocessor.get_feature_names_out())

    X_test_pp = preprocess_new_data(X_test, preprocessor=preprocessor)
    X_test_pp = pd.DataFrame(X_test_pp, columns=preprocessor.get_feature_names_out())

    X_train_sequences, y_train_sequences, X_test_sequences, y_test_sequences = preproc_rnn(X_train_pp=X_test_pp=,
                                                                                           X_test_pp=X_test_pp, quarters_input=quarters_input)

    adam = Adam(learning_rate=0.002, beta_1=0.95)
    weight_for_0 = (0.4 / (len(y_train_sequences) - sum(y_train_sequences))) * (len(y_train_sequences) / 2)
    weight_for_1 = (0.6 / sum(y_train_sequences)) * (len(y_train_sequences) / 2)

    es = EarlyStopping(patience=7, restore_best_weights=True)
    plateau = ReduceLROnPlateau()
    class_weight = {0: weight_for_0, 1: weight_for_1}

    # 1- RNN Architecture
    model = Sequential()
    model.add(layers.LSTM(units=240, activation='tanh', input_shape=(quarters_input, 121), return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(units=160, activation='tanh'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(140, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(60, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(30, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation=final_activation))

    # 2- Compilation
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=metrics)

    # 3- Fit
    model.fit(X_train_sequences, y_train_sequences, validation_split=0.2, epochs=100, batch_size=32,
                        callbacks=[es, plateau], verbose=3)

    return model

def save_rnn(model, quarters_input, threshold, small_cap):
    # Save Model
    file_name = f"{datetime.datetime.now()}_RNN_{quarters_input}_qtr_{threshold}_ths_sc_{small_cap}.pkl"
    model_dir = '../models/'

    with open(model_dir+file_name, "wb") as file:
        pickle.dump(model, file)

if __name__ == '__main__':
    df = pd.read_csv('raw_data/data_for_preprocessing.csv', index_col=0)
    model = run_RNN(df)
    print(model.summary())
