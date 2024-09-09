# Importing all the good stuff
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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

def drop_columns(df, cols_to_drop=None):
    """Drops specified columns from the DataFrame."""
    if cols_to_drop is None:
        # Default columns to drop if none are specified
        cols_to_drop = ['cik', 'CIK', 'date', 'stprba', 'quarter', 'year']
    return df.drop(cols_to_drop, axis=1, errors='ignore')


# Creating a custom function for the group split
def group_train_test_split(data, test_size=0.2, random_state=None):
    # We split by groups (company ticker) while keeping the data structure intact.
    unique_groups = data['TICKER'].unique()
    train_groups, test_groups = train_test_split(unique_groups, test_size=test_size, random_state=random_state)

    # Split into train and test sets
    X_train = data[data['TICKER'].isin(train_groups)]
    X_test = data[data['TICKER'].isin(test_groups)]

    # Define columns to drop: Ticker, cik, date, quarter, year + growth columns
    cols_to_drop = ['mc_qtr_growth', 'mc_qtr_growth_pct', 'mc_yr_growth', 'mc_yr_growth_pct', 'mc_2yr_growth', 'mc_2yr_growth_pct']

    # Drop unwanted columns
    X_train = drop_columns(X_train, cols_to_drop + ['cik', 'CIK', 'date', 'stprba', 'quarter', 'year'])
    X_test = drop_columns(X_test, cols_to_drop + ['cik', 'CIK', 'date', 'stprba', 'quarter', 'year'])

    # Extract the target variable from the dataset
    y_train = data[data['TICKER'].isin(train_groups)]['target']
    y_test = data[data['TICKER'].isin(test_groups)]['target']

    return X_train, X_test, y_train, y_test

# Step 2: Identify numerical and categorical features
def identify_feature_types(df):
    """Identifies the numerical and categorical columns in the DataFrame."""
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Exclude 'Ticker' from categorical features as it's not needed for transformation
    if 'TICKER' in categorical_features:
        categorical_features.remove('TICKER')

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

# Step 5: Function to preprocess new/unseen/test data in production mode (only transforming)
def preprocess_new_data(X_new, preprocessor):
    """Transforms new/unseen/test data using a pre-fitted pipeline."""
    if preprocessor is None:
        raise ValueError("The preprocessor must be fitted on training data first before transforming new data.")

    # Transform the new data (no fitting here)
    X_new_processed = preprocessor.transform(X_new)
    return X_new_processed
