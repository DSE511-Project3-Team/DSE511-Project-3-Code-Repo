# Script to run all the code necessary to generate the results

from src.data.download_data import generate_base_data
from src.preprocessing.process_imputed_data import generate_imputed_data
from src.preprocessing.process_modelling_data import get_modelling_data
from src.models.tune_xgboost import perform_xgboost_tuning
from src.models.tune_adaboost import perform_adaboost_tuning 
from src.results.xgboost_results import get_xgboost_results
from src.results.adaboost_results import get_adaboost_results

## DATA PREPROCESSING

# 1. Download and filter data for six cities
# generate_base_data()

# 2. Impute the missing values 
# generate_imputed_data()

# 3. Perform feature encoding, extraction, standardization, and train/val/test split
X = get_modelling_data()
X_train, X_val, X_test, X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test = X


## HYPERPARAMETER TUNING

# 1. Tune XGBoost
# perform_xgboost_tuning(X, 'full')
# perform_xgboost_tuning(X, 'pca')

# 2. Tune AdaBoost
# perform_adaboost_tuning(X, 'full')
# perform_adaboost_tuning(X, 'pca')

## RESULTS

# 1. XGBoost Results
# X_ = X.copy()
# get_xgboost_results(X)

# 2. AdaBoost Results
get_adaboost_results(X)