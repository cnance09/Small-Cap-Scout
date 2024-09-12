import argparse
import pandas as pd
import pickle
import os
from tqdm import tqdm
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV

from xgboost import XGBClassifier

def get_model(model_type):
    if model_type == 'logistic_regression':
        return LogisticRegression(solver='saga', max_iter=2000)
    elif model_type == 'knn':
        return KNeighborsClassifier()
    elif model_type == 'svc':
        return SVC()
    elif model_type == 'mlpclassifier':
        return MLPClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and save model')
    parser.add_argument('--model_type', type=str, required=True, help='Model type to train (e.g. logistic_regression, knn, svc, mlpclassifier)')
    args = parser.parse_args()

    model = get_model(args.model_type)

def save_model(model, model_type, model_dir='~/models/'):
    """Saves the trained model with a timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_filename = f'{model_type}_{timestamp}.pkl'

    # Ensure model directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the trained model
    model_path = os.path.join(model_dir, model_filename)
    with open(model_path, 'wb') as f_model:
        pickle.dump(model, f_model)

    print(f"Model saved to: {model_path}")
    return model_path

def evaluate_model(model, X_train, y_train, X_test, y_test, scoring_metrics=['accuracy', 'precision', 'recall', 'f1']):
    """Evaluates the model with cross-validation and test set metrics."""
    cv_metrics = {}
    for metric in scoring_metrics:
        with tqdm(total=5, desc=f"Cross-Validation ({metric})", bar_format='{l_bar}{bar} [elapsed: {elapsed} left: {remaining}]') as pbar:
            cv_metrics[metric] = cross_val_score(model, X_train, y_train, cv=5, scoring=metric)
            pbar.update(5)

    print(f"Cross-validated Metrics: {', '.join([f'{m}: {cv_metrics[m].mean():.4f}' for m in cv_metrics])}")

    # Test on the test set
    y_pred_test = model.predict(X_test)

    # Calculate test set metrics
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test),
        'recall': recall_score(y_test, y_pred_test),
        'f1': f1_score(y_test, y_pred_test)
    }

    # Combine cross-validated and test metrics
    metrics = {**{f'cv_{m}': cv_metrics[m].mean() for m in cv_metrics}, **test_metrics}
    return metrics

def train_logistic_regression_and_save(X_train, y_train, X_test, y_test, model_dir='~/models/'):
    """Trains, evaluates a logistic regression model, saves the trained model, and returns evaluation metrics."""

    model_type = 'logistic_regression'
    model = LogisticRegression(C=0.001, max_iter=2000, solver='lbfgs')

    # Train model with a progress bar
    with tqdm(total=100, desc=f"Training {model_type}", bar_format='{l_bar}{bar} [elapsed: {elapsed} left: {remaining}]') as pbar:
        model.fit(X_train, y_train)
        pbar.update(100)

    # Check number of iterations
    print(f"Number of iterations: {model.n_iter_}")

    # Evaluate the model
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Save the model
    save_model(model, model_type, model_dir)

    return metrics, model

def train_knn_and_save(X_train, y_train, X_test, y_test, model_dir='~/models/'):
    """Trains, evaluates a K-Nearest Neighbors model, saves the trained model, and returns evaluation metrics."""

    model_type = 'knn'
    knn = KNeighborsClassifier()

    # Train model with a progress bar
    with tqdm(total=100, desc=f"Training {model_type}", bar_format='{l_bar}{bar} [elapsed: {elapsed} left: {remaining}]') as pbar:
        knn.fit(X_train, y_train)
        pbar.update(100)

    # Evaluate the model
    metrics = evaluate_model(knn, X_train, y_train, X_test, y_test)

    # Save the model
    save_model(knn, model_type, model_dir)

    return metrics, knn

def train_svc_rbf_and_save(X_train, y_train, X_test, y_test, model_dir='~/models/'):
    """Trains, evaluates an SVM with RBF kernel, saves the trained model, and returns evaluation metrics."""

    model_type = 'svc_rbf'
    svc_rbf = SVC(kernel='rbf', probability=True)  # Set `probability=True` for log_loss and cross-validation

    # Train model with a progress bar
    #with tqdm(total=100, desc=f"Training {model_type}", bar_format='{l_bar}{bar} [elapsed: {elapsed} left: {remaining}]') as pbar:
    svc_rbf.fit(X_train, y_train)
        #pbar.update(100)

    # Evaluate the model
    metrics = evaluate_model(svc_rbf, X_train, y_train, X_test, y_test)

    # Save the model
    save_model(svc_rbf, model_type, model_dir)

    return metrics, svc_rbf


def train_xgb_and_save(X_train, y_train, X_test, y_test, model_dir='~/models/'):
    """Trains, evaluates an XGBClassifier model, saves the trained model, and returns evaluation metrics."""

    model_type = 'xgb_classifier'

    # Initialize XGBClassifier with specified parameters and verbose for progress monitoring
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=5, verbose=1)

    # Train the model
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

    # Evaluate the model
    metrics = evaluate_model(xgb, X_train, y_train, X_test, y_test)

    # Save the model
    save_model(xgb, model_type, model_dir)

    return metrics, xgb


def run_grid_search(X_train, y_train):
    """Runs a grid search on logistic regression model to find the best hyperparameters."""

    # Define the parameter grid for Logistic Regression
    param_grid = {
        'solver': ['saga', 'lbfgs'],  # Different solvers
        'max_iter': [1500, 2000, 2500],  # Number of iterations
        'C': [0.005, 0.007, 0.01, 0.12, 0.15]  # Regularization strength
    }

    # Create a Logistic Regression model
    logistic_model = LogisticRegression()

    # Set up the GridSearchCV
    grid_search = GridSearchCV(
        estimator=logistic_model,
        param_grid=param_grid,
        scoring='precision',  # Choose appropriate scoring metric
        cv=5,  # Number of cross-validation folds
        n_jobs=-1,  # Use all available cores
        verbose=1  # Verbosity level
    )

    # Fit Grid Search
    grid_search.fit(X_train, y_train)

    # Best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")

    # Get the best model
    best_model = grid_search.best_estimator_

    return best_model, best_params, best_score

def knn_run_grid_search(X_train, y_train):
    """Runs a grid search on knn model to find the best hyperparameters."""

    # Define the parameter grid for Logistic Regression
    param_grid = {
        'n_neighbors': [550],  # Different solvers
        #'max_iter': [1500, 2000, 2500],  # Number of iterations
        #'C': [0.005, 0.007, 0.01, 0.12, 0.15]  # Regularization strength
    }

    # Create a Logistic Regression model
    knn_model = KNeighborsClassifier()

    # Set up the GridSearchCV
    grid_search = GridSearchCV(
        estimator=knn_model,
        param_grid=param_grid,
        scoring='precision',  # Choose appropriate scoring metric
        cv=3,  # Number of cross-validation folds
        n_jobs=-1,  # Use all available cores
        verbose=1  # Verbosity level
    )

    # Fit Grid Search
    grid_search.fit(X_train, y_train)

    # Best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")

    # Get the best model
    best_model = grid_search.best_estimator_

    return best_model, best_params, best_score


def run_xgboost(X_train, y_train):
    """Runs a grid search on knn model to find the best hyperparameters."""

    # Define the parameter grid for Logistic Regression
    param_grid = param_grid = {'n_estimators': [200],
                               'learning_rate': [0.01],
                               'max_depth': [5]}

#{'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 200}

    # Create a Logistic Regression model
    model = XGBClassifier()

    # Set up the search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='precision',  # Choose appropriate scoring metric
        cv=3,  # Number of cross-validation folds
        n_jobs=-1,  # Use all available cores
        verbose=1  # Verbosity level
    )

    # Fit Grid Search
    grid_search.fit(X_train, y_train)

    # Best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")

    # Get the best model
    best_model = grid_search.best_estimator_

    return best_model, best_params, best_score
