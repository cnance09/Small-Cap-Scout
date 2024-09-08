import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Step 1 : target creation + train_test_split
# Creating target variables to automate creation of quarterly, yearly and 2-yearly targets, because well, DON'T REPEAT YOURSELF!
def create_target_variable(df, frequency:int, threshold):
    if frequency == 1:
        col = 'mc_qtr_growth_pct'
    if frequency == 4:
        col = 'mc_yr_growth_pct'
    if frequency == 8:
        col = 'mc_2yr_growth_pct'
   #else:
   #    raise ValueError("Invalid frequency. Use 1 (quarterly), 4 (yearly), or 8 (2-year).")
    df[col] = df[col].shift(-frequency)
    df.dropna(subset=col, inplace=True)
    target_func = lambda x: 1 if ((x[col] > threshold) & (x.small_cap == 1)) else 0
    df['target'] = df.apply(target_func, axis=1)
    return df

# Creating a custom function for the group split
def group_train_test_split(data, test_size=0.2, random_state=None):
    # We split by groups (company ticker) while keeping the data structure intact.
    unique_groups = data['Ticker'].unique()
    train_groups, test_groups = train_test_split(unique_groups, test_size=test_size, random_state=random_state)
    X_train = data[data['Ticker'].isin(train_groups)].drop(['mc_qtr_growth', 'mc_qtr_growth_pct', 'mc_yr_growth', 'mc_yr_growth_pct', 'mc_2yr_growth', 'mc_2yr_growth_pct'], axis = 1)
    X_test = data[data['Ticker'].isin(test_groups)].drop(['mc_qtr_growth', 'mc_qtr_growth_pct', 'mc_yr_growth', 'mc_yr_growth_pct', 'mc_2yr_growth', 'mc_2yr_growth_pct'], axis = 1)
    y_train = data[data['Ticker'].isin(train_groups)]['target']
    y_test = data[data['Ticker'].isin(test_groups)]['target']
    return X_train, X_test, y_train, y_test

# Step 2: Identify numerical and categorical features
def identify_feature_types(df):
    """Identifies the numerical and categorical columns in the DataFrame."""
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Exclude 'Ticker' from categorical features as it's not needed for transformation
    if 'Ticker' in categorical_features:
        categorical_features.remove('Ticker')

    return numerical_features, categorical_features

# Step 3: Create preprocessing pipeline for numerical and categorical features
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
            ('cat', categorical_transformer, categorical_features),
            ('drop_date', 'drop', ['date'])  # Ensure to drop or transform date columns
        ]
    )

    return preprocessor

# Step 4: Function to preprocess data in training mode (fitting the pipeline)
def preprocess_training_data(X_train, preprocessor=None):
    """Fits and transforms the training data using the provided pipeline."""
    if preprocessor is None:
        # Identify feature types
        numerical_features, categorical_features = identify_feature_types(X_train)
        preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features)

    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    return X_train_processed, preprocessor

# Step 5: Function to preprocess new/unseen/test data in production mode (only transforming)
def preprocess_new_data(X_new, preprocessor):
    """Transforms new/unseen/test data using a pre-fitted pipeline."""
    if preprocessor is None:
        raise ValueError("The preprocessor must be fitted on training data first before transforming new data.")

    # Transform the new data (no fitting here)
    X_new_processed = preprocessor.transform(X_new)
    return X_new_processed

# Step 6: Function to predict based on different target inputs defined at the create_target_variable stage: quarterly (frequency=1), yearly (frequency=4), and 2-year (frequency=8) predictions
def train_logistic_regression(X_train, y_train, X_test, y_test):
    """ Trains and evaluates a logistic regression model, and returns multiple evaluation metrics
    (accuracy, precision, recall, F1-score) using cross-validation and test data.
    """
    # Train logistic regression model with a progress bar
    logistic_model = LogisticRegression(solver='saga', max_iter=5000)

    # Display progress during model fitting
    with tqdm(total=100, desc="Training Logistic Regression", bar_format='{l_bar}{bar} [elapsed: {elapsed} left: {remaining}]') as pbar:
        logistic_model.fit(X_train, y_train)
        pbar.update(100)  # Update the progress bar after training completes

    # Check number of iterations
    print(f"Number of iterations: {logistic_model.n_iter_}")

    # Evaluate using cross-validation for accuracy, precision, recall, and F1-score with progress
    cv_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        with tqdm(total=5, desc=f"Cross-Validation ({metric})", bar_format='{l_bar}{bar} [elapsed: {elapsed} left: {remaining}]') as pbar:
            cv_metrics[metric] = cross_val_score(logistic_model, X_train, y_train, cv=5, scoring=metric)
            pbar.update(5)

    # Print cross-validation scores
    print(f"Cross-validated Metrics: {', '.join([f'{m}: {cv_metrics[m].mean():.4f}' for m in cv_metrics])}")

    # Test on the test set
    y_pred_test = logistic_model.predict(X_test)

    # Calculate test set metrics
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test),
        'recall': recall_score(y_test, y_pred_test),
        'f1': f1_score(y_test, y_pred_test)
    }

    # Create a dictionary to store all metrics
    metrics = {**{f'cv_{m}': cv_metrics[m].mean() for m in cv_metrics}, **test_metrics}

    return metrics, logistic_model


'''# HOW TO USE in terminal:

cd ~/Small-Cap-Scout/smallscout
ipython
import pandas as pd
from preprocessor import create_target_variable, group_train_test_split, identify_feature_types, create_preprocessing_pipeline,preprocess_training_data, preprocess_new_data,train_logistic_regression

# Step 1: Target creation and splitting
df = pd.read_csv('~/Small-Cap-Scout/raw_data/merged_data_prelim_stocks.csv')
df_qtr = create_target_variable(df, frequency=1, threshold=0.5)
df_yr = create_target_variable(df, frequency=4, threshold=0.5)
df_2yr = create_target_variable(df, frequency=8, threshold=0.5)

X_train_qtr, X_test_qtr, y_train_qtr, y_test_qtr = group_train_test_split(df_qtr)
X_train_yr, X_test_yr, y_train_yr, y_test_yr = group_train_test_split(df_yr)
X_train_2yr, X_test_2yr, y_train_2yr, y_test_2yr = group_train_test_split(df_2yr)

# Step 2: Identify feature types after splitting
numerical_features_qtr, categorical_features_qtr = identify_feature_types(X_train_qtr)
numerical_features_yr, categorical_features_yr = identify_feature_types(X_train_yr)
numerical_features_2yr, categorical_features_2yr = identify_feature_types(X_train_2yr)

# Step 3: Create the preprocessing pipeline
preprocessor_qtr = create_preprocessing_pipeline(numerical_features_qtr, categorical_features_qtr)
preprocessor_yr = create_preprocessing_pipeline(numerical_features_yr, categorical_features_yr)
preprocessor_2yr = create_preprocessing_pipeline(numerical_features_2yr, categorical_features_2yr)

# Step 4: Preprocess the training data
X_train_qtr_processed, preprocessor_qtr = preprocess_training_data(X_train_qtr, preprocessor=preprocessor_qtr)
X_train_yr_processed, preprocessor_yr = preprocess_training_data(X_train_yr, preprocessor=preprocessor_yr)
X_train_2yr_processed, preprocessor_2yr = preprocess_training_data(X_train_2yr, preprocessor=preprocessor_2yr)

# Step 5 : then the test data
X_test_qtr_processed = preprocess_new_data(X_test_qtr, preprocessor_qtr)
X_test_yr_processed = preprocess_new_data(X_test_yr, preprocessor_yr)
X_test_2yr_processed = preprocess_new_data(X_test_2yr, preprocessor_2yr)

# Train for quarterly (frequency=1), yearly (frequency=4), and 2-year (frequency=8) predictions
y_pred_qtr, model_qtr = train_logistic_regression(X_train_qtr_processed, y_train_qtr, X_test_qtr_processed, y_test_qtr)
y_pred_yr, model_yr = train_logistic_regression(X_train_yr_processed, y_train_yr, X_test_yr_processed, y_test_yr)
y_pred_2yr, model_2yr = train_logistic_regression(X_train_2yr_processed, y_train_2yr, X_test_2yr_processed, y_test_2yr)

# Print metrics for each model
print("1 Quarter Ahead Metrics:", y_pred_qtr)
print("1 Year Ahead Metrics:", y_pred_yr)
print("2 Years Ahead Metrics:", y_pred_2yr)'''





''' def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a logistic regression model, and returns multiple evaluation metrics
    (accuracy, precision, recall, F1-score) using cross-validation and test data.
    """
    # Train logistic regression model
    logistic_model = LogisticRegression(solver='saga', max_iter=2000)
    logistic_model.fit(X_train, y_train)

    # Evaluate using cross-validation for accuracy, precision, recall, and F1-score
    cv_accuracy = cross_val_score(logistic_model, X_train, y_train, cv=5, scoring='accuracy')
    cv_precision = cross_val_score(logistic_model, X_train, y_train, cv=5, scoring='precision')
    cv_recall = cross_val_score(logistic_model, X_train, y_train, cv=5, scoring='recall')
    cv_f1 = cross_val_score(logistic_model, X_train, y_train, cv=5, scoring='f1')

    # Print cross-validation scores
    print(f"Cross-validated Accuracy: {cv_accuracy.mean():.4f}")
    print(f"Cross-validated Precision: {cv_precision.mean():.4f}")
    print(f"Cross-validated Recall: {cv_recall.mean():.4f}")
    print(f"Cross-validated F1-Score: {cv_f1.mean():.4f}")

    # Test on the test set
    y_pred_test = logistic_model.predict(X_test)

    # Calculate test set metrics
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)

    # Create a dictionary to store all metrics
    metrics = {
        'cv_accuracy': cv_accuracy.mean(),
        'cv_precision': cv_precision.mean(),
        'cv_recall': cv_recall.mean(),
        'cv_f1': cv_f1.mean(),
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    }

    return metrics, logistic_model'''
