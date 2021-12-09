import copy 
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score 
from prettytable import PrettyTable
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier


def perform_random_forest_testing(X, param):
    X_train, X_val, X_test, X_train_pca, X_val_pca, \
                X_test_pca, y_train, y_val, y_test = X
    
    RF = RandomForestClassifier(random_state = 44)
    
    if param == 'full':
        print("=========================================")
        print(f"\033[1m Random Forest Testing Results on Full Data\033[0m")
        print("=========================================")
        
        xtrain, xval, xtest = X_train, X_val, X_test
        
        best_param_dict = {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'auto',
                           'max_depth': None, 'criterion': 'gini'}
        
    elif param == 'pca':
        
        print("=========================================")
        print(f"\033[1m Random Forest Testing Results on PCA Data\033[0m")
        print("=========================================")
        
        xtrain, xval, xtest = pd.DataFrame(X_train_pca), pd.DataFrame(X_val_pca), pd.DataFrame(X_test_pca)
        
        best_param_dict = {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'auto',
                           'max_depth': None, 'criterion': 'gini'}
        
    start = time.time()
    
    best_model = RandomForestClassifier(random_state = 44,
                                n_estimators = best_param_dict['n_estimators'],
                                max_depth = best_param_dict['max_depth'],
                                min_samples_split = best_param_dict['min_samples_split'], 
                                min_samples_leaf = best_param_dict['min_samples_leaf'],
                                criterion = best_param_dict['criterion'],
                                max_features = best_param_dict['max_features']).fit(pd.concat([xtrain, xval]), pd.concat([y_train, y_val]))
    
    stop = time.time()
        
    time_to_complete = stop - start
    
    preds = best_model.predict(xtest)

    conf_mat = confusion_matrix(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    

    pt = PrettyTable(['Time to Test (s)', 'Accuracy', 'Sensitivity', 
                          'Specificity', 'Precision', 'F1 Score (macro)']) 

    pt.add_row([round(time_to_complete, 2), round(accuracy, 2), 
                round(conf_mat[1,1]/(conf_mat[1,1]+conf_mat[1,0]), 2),
                round(conf_mat[0,0]/(conf_mat[0,1]+conf_mat[0,0]), 2),
                round(conf_mat[1,1]/(conf_mat[1,1]+conf_mat[0,1]), 2),
                round(f1_score(y_test, preds, average = 'macro'), 2)])
    
    conf_mat1 = PrettyTable(['Confusion Matrix','Predicted Pos.', 'Predicted Neg.'])
    conf_mat1.add_row(['Actual Postive', conf_mat[1,1], conf_mat[1,0]])
    conf_mat1.add_row(['Actual Negative', conf_mat[0,1], conf_mat[0,0]])

    print(conf_mat1)
    
    print('RANDOM FOREST BEST MODEL BASED ON TESTING DATA:')
    
    print(pt)
    
    print('The best parameters are: ', best_param_dict)
    
    return (preds, best_model)
    
