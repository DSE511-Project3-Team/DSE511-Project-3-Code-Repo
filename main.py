# Script to run all the necessary code to generate the results

import sys
from src.data.download_data import generate_base_data
from src.preprocessing.process_imputed_data import generate_imputed_data
from src.preprocessing.process_modelling_data import get_modelling_data
<<<<<<< HEAD
from src.models.tune_xgboost import perform_xgboost_tuning, perform_xgboost_tuning_2
=======

from src.models.tune_xgboost import perform_xgboost_tuning
>>>>>>> main
from src.models.tune_adaboost import perform_adaboost_tuning 
from src.models.tune_logistic_regression import perform_Logistic_Regression_tuning 
from src.models.tune_random_forest import perform_random_forest_tuning 
from src.models.tune_multinomial_bayes import perform_multinomial_naive_bayes_tuning 
from src.models.tune_gradient_boosting_classifier import perform_gradient_boosting_tuning 

from src.results.xgboost_results import get_xgboost_results
from src.results.adaboost_results import get_adaboost_results
from src.results.logistic_regression_results import perform_Logistic_Regression_testing
from src.results.random_forest_results import perform_random_forest_testing
from src.results.multinomial_bayes_results import perform_multinomial_naive_bayes_testing
from src.results.gradient_boosting_results import perform_gradient_boosting_testing

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

    # Tune regularization parameter
    perform_xgboost_tuning_2(X)

    # 2. Tune AdaBoost
    perform_adaboost_tuning(X, 'full')
    perform_adaboost_tuning(X, 'pca')
    
    # 3. Tune Logistic Regression
    perform_Logistic_Regression_tuning(X, 'full')
    perform_Logistic_Regression_tuning(X, 'pca')

    # 4. Tune Random Forest
    perform_random_forest_tuning(X, 'full')
    perform_random_forest_tuning(X, 'pca')
    
    # 5. Tune Naive Bayes
    perform_multinomial_naive_bayes_tuning(X, 'full')
    perform_multinomial_naive_bayes_tuning(X, 'pca')

    # 6. Tune Gradient Boosting Classifier
    perform_gradient_boosting_tuning(X, 'full')
    perform_gradient_boosting_tuning(X, 'pca')
    
def generate_results():
    # Perform feature encoding, extraction, standardization, and train/val/test split
    X = get_modelling_data()
    X_train, X_val, X_test, X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test = X

    # 1. XGBoost Results
    get_xgboost_results(X)

    # 2. AdaBoost Results
    get_adaboost_results(X)
    
    # 3. Logistic Regression Results
    perform_Logistic_Regression_testing(X=X, param='full')
    perform_Logistic_Regression_testing(X=X, param='pca')

    # 4. Random Forest Results
    perform_random_forest_testing(X=X, param='full')
    perform_random_forest_testing(X=X, param='pca')
    
    # 5. Naive Bayes Results
    perform_multinomial_naive_bayes_testing(X=X, param='full')
    perform_multinomial_naive_bayes_testing(X=X, param='pca')

    # 6. Gradient Boosting Classifier Results
    perform_gradient_boosting_testing(X=X, param='full')
    perform_gradient_boosting_testing(X=X, param='pca')
    
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
