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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

def identify_feature_types(df):
    """Identifies the numerical and categorical columns in the DataFrame."""
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Exclude 'Ticker' from categorical features as it's not needed for transformation
    if 'target' in numerical_features:
        numerical_features.remove('target')
    if 'cik' in numerical_features:
        numerical_features.remove('cik')
    if 'TICKER' in categorical_features:
        categorical_features.remove('TICKER')
    if 'quarter' in categorical_features:
        categorical_features.remove('quarter')

    return numerical_features, categorical_features

def create_preprocessing_pipeline(numerical_features, categorical_features):
    """Creates the preprocessing pipeline for numerical and categorical features."""
    # Preprocessing for numerical data: RobustScaler to make our numbers más robusto.
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle NaNs
        ('scaler', RobustScaler())  # Scale the data
    ])

    # Preprocessing for categorical data: OneHotEncoder to give each category its own columm...
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing categories
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Encode categories
    ])

    # Combine the transformers into one big ColumnTransformer.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    return preprocessor

def preprocess_training_data(X_train, preprocessor=None):
    """Fits and transforms the training data using the provided pipeline."""
    if preprocessor is None:
        # Identify feature types
        numerical_features, categorical_features = identify_feature_types(X_train)
        preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features)

    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    return X_train_processed, preprocessor

def preprocess_new_data(X_new, preprocessor):
    """Transforms new/unseen/test data using a pre-fitted pipeline."""
    if preprocessor is None:
        raise ValueError("The preprocessor must be fitted on training data first before transforming new data.")

    # Transform the new data (no fitting here)
    X_new_processed = preprocessor.transform(X_new)
    return X_new_processed

def run_RNN(df, quarters_input=4, threshold=0.25, small_cap=True, model_type='classifier', horizon='quarter'):

    # Set model according to given parameters
    if horizon == 'quarter':
        col = 'mc_qtr_growth_pct'
        adjustment = 0
    elif horizon == 'year':
        col = 'mc_yr_growth_pct'
        adjustment = 3
    elif horizon == 'year':
        col = 'mc_2yr_growth_pct'
        adjustment = 7

    if model_type == 'classifier':
        final_activation = 'sigmoid'
        metrics=['accuracy', 'precision', 'recall']
    else:
        final_activation = 'linear'
        metrics=['r2', 'mse', 'mae']

    if small_cap==True:
        target_func = lambda x: 1 if ((x[col] > threshold) & (x.small_cap == 1)) else 0
    else:
        target_func = lambda x: 1 if ((x[col] > threshold)) else 0

    if model_type == 'classifier':
        df['target'] = df.apply(target_func, axis=1)
    else:
        df['target'] = df[col]

    # Train_Test Split
    unique_groups = df['TICKER'].unique()
    train_groups, test_groups = train_test_split(unique_groups, test_size=0.3, random_state=42)

    data_train = df[df['TICKER'].isin(train_groups)]
    data_test = df[df['TICKER'].isin(test_groups)]

    # Remove columns
    cols_drop = df.columns.tolist()[-12:]
    cols_drop.remove('TICKER')
    cols_drop.remove('small_cap')
    cols_drop.remove('micro_cap')
    cols_drop.remove('target')
    cols_drop += ['date', 'year']

    X_train = data_train[data_train['TICKER'].isin(train_groups)].drop(columns=cols_drop).reset_index(drop=True)
    X_test = data_test[data_test['TICKER'].isin(test_groups)].drop(columns=cols_drop).reset_index(drop=True)

    # Preprocess X_train and X_test
    num, cat = identify_feature_types(X_train)
    preprocessor = create_preprocessing_pipeline(num, cat)

    X_train_pp, preprocessor = preprocess_training_data(X_train, preprocessor=preprocessor)
    X_train_pp = pd.DataFrame(X_train_pp, columns=preprocessor.get_feature_names_out())

    X_test_pp = preprocess_new_data(X_test, preprocessor=preprocessor)
    X_test_pp = pd.DataFrame(X_test_pp, columns=preprocessor.get_feature_names_out())

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

    model = Sequential()
    model.add(layers.LSTM(units=80, activation='tanh', input_shape=(quarters_input, 116)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(40, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation=final_activation))

    # 2- Compilation
    model.compile(loss='binary_crossentropy',
                optimizer=Adam(learning_rate=0.002, beta_1=0.75),
                metrics=metrics)

    # 3- Fit
    es = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(X_train_sequences, y_train_sequences, validation_split=0.2, epochs=100, batch_size=32,
                        callbacks=[es], verbose=3)

    # Save Model
    file_name = f"{datetime.datetime.now()}_RNN_{model_type}_{quarters_input}_qtr_{threshold}_ths_sc_{small_cap}.pkl"
    model_dir = 'models/'

    with open(model_dir+file_name, "wb") as file:
        pickle.dump(model, file)

    return model

df = pd.read_csv('raw_data/data_for_preprocessing.csv', index_col=0)
model = run_RNN(df)
print(model.summary())
