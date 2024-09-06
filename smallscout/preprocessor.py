import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Step 1: Custom train/test split by grouping data by 'Ticker'
def group_train_test_split(X, test_size=0.2, random_state=None):
    """Splits the data into train/test sets, grouping by 'Ticker'."""
    unique_groups = X['Ticker'].unique()
    train_groups, test_groups = train_test_split(unique_groups, test_size=test_size, random_state=random_state)
    train_data = X[X['Ticker'].isin(train_groups)]
    test_data = X[X['Ticker'].isin(test_groups)]
    return train_data, test_data

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
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle NaNs
        ('scaler', RobustScaler())  # Scale the data
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing categories
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Encode categories
    ])

    # Combine the transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
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

# Step 5: Function to preprocess new data in production mode (only transforming)
def preprocess_new_data(X_new, preprocessor):
    """Transforms new/unseen data using a pre-fitted pipeline."""
    if preprocessor is None:
        raise ValueError("The preprocessor must be fitted on training data first before transforming new data.")

    # Transform the new data (no fitting here)
    X_new_processed = preprocessor.transform(X_new)
    return X_new_processed

# Example of usage in both training and production:

# Load dataset
df = data_set  # Assuming 'dataset' is already a pandas DataFrame

# Step 6: Perform group-based train/test split (during training phase)
train_data, test_data = group_train_test_split(df, test_size=0.2, random_state=42)

# Step 7: Separate the target variable
y_train = train_data['Monthly Avg Market Cap']
X_train = train_data.drop('Monthly Avg Market Cap', axis=1)

y_test = test_data['Monthly Avg Market Cap']
X_test = test_data.drop('Monthly Avg Market Cap', axis=1)

# Step 8: Preprocess training data (fit and transform)
X_train_processed, preprocessor = preprocess_training_data(X_train)

# Step 9: Preprocess test data (transform only, simulating a production scenario)
X_test_processed = preprocess_new_data(X_test, preprocessor)

# Step 10: In a real production environment, you would apply the following on unseen data:
# new_data = ... (load new data here)
# X_new_processed = preprocess_new_data(new_data, preprocessor)
