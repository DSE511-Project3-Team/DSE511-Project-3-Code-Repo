import copy 
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score 
from prettytable import PrettyTable
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def perform_gradient_boosting_testing(X, param):
    
    X_train, X_val, X_test, X_train_pca, X_val_pca, \
                X_test_pca, y_train, y_val, y_test = X
    
    
    clf = GradientBoostingClassifier(random_state=44)
    
    if param == 'full':
        print("=========================================")
        print(f"\033[1m Gradient Boosting Testing Results on Full Data\033[0m")
        print("=========================================")
        
        xtrain, xval, xtest= pd.DataFrame(X_train), pd.DataFrame(X_val), pd.DataFrame(X_test)
        best_param_dict = {'n_estimators': 366, 'max_features': 'log2', 'max_depth': 16, 'loss': 'deviance',
         'learning_rate': 0.9134564564564565, 'criterion': 'mse'}
        
    elif param == 'pca':
        
        print("=========================================")
        print(f"\033[1m Gradient Boosting Testing Results on PCA Data\033[0m")
        print("=========================================")
        
        xtrain, xval, xtest = pd.DataFrame(X_train_pca), pd.DataFrame(X_val_pca), pd.DataFrame(X_test_pca)
        best_param_dict = {'n_estimators': 366, 'max_features': 'log2', 'max_depth': 16, 'loss': 'deviance',
         'learning_rate': 0.9134564564564565, 'criterion': 'mse'}  
        
    start = time.time()

    best_model = GradientBoostingClassifier(random_state=44,
                                            loss = best_param_dict['loss'],
                                            n_estimators = best_param_dict['n_estimators'],
                                            learning_rate = best_param_dict['learning_rate'],
                                            max_depth = best_param_dict['max_depth'],
                                            max_features = best_param_dict['max_features'],
                                            criterion = best_param_dict['criterion']).fit(pd.concat([xtrain, xval]), pd.concat([y_train, y_val]))   
    
    preds = best_model.predict(xtest)
    
    stop = time.time()
    time_to_complete = stop - start

    conf_mat = confusion_matrix(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    

    pt = PrettyTable(['Time to Tune (s)', 'Accuracy', 'Sensitivity', 
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
    
    print('GRADIENT BOOSTING BEST MODEL BASED ON TESTING:')
    
    print(pt)
    
    print('The parameters used were: ', best_param_dict)
    
    return (preds, best_model)
