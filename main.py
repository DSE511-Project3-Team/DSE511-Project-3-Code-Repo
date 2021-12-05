# Script to run all the necessary code to generate the results

import sys
from src.data.download_data import generate_base_data
from src.preprocessing.process_imputed_data import generate_imputed_data
from src.preprocessing.process_modelling_data import get_modelling_data
from src.models.tune_xgboost import perform_xgboost_tuning
from src.models.tune_adaboost import perform_adaboost_tuning 
from src.results.xgboost_results import get_xgboost_results
from src.results.adaboost_results import get_adaboost_results

def generate_data():
    # Download, filter and impute data for six cities
    generate_base_data()
    generate_imputed_data()

def perform_tuning():
    # Perform feature encoding, extraction, standardization, and train/val/test split
    X = get_modelling_data()
    X_train, X_val, X_test, X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test = X

    # 1. Tune XGBoost
    perform_xgboost_tuning(X, 'full')
    perform_xgboost_tuning(X, 'pca')

    # 2. Tune AdaBoost
    perform_adaboost_tuning(X, 'full')
    perform_adaboost_tuning(X, 'pca')

def generate_results():
    # Perform feature encoding, extraction, standardization, and train/val/test split
    X = get_modelling_data()
    X_train, X_val, X_test, X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test = X

    # 1. XGBoost Results
    get_xgboost_results(X)

    # 2. AdaBoost Results
    get_adaboost_results(X)

if __name__ == "__main__":

    ## 1. GENERATE DATA 
    if sys.argv[1] == 'data':
        print("Generating Dataset ... ")
        generate_data()

    ## 2. HYPERPARAMETER TUNING
    if sys.argv[1] == 'tune':
        print("Performing Hyperparameter Tuning ... ")
        perform_tuning()

    ## 3. RESULTS
    if sys.argv[1] == 'results':
        print("Generating Results ... ")
        generate_results()
