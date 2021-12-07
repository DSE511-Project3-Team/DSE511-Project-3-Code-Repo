import copy 
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score 
from prettytable import PrettyTable
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def perform_Logistic_Regression_testing(X, param):
    X_train, X_val, X_test, X_train_pca, X_val_pca, \
                X_test_pca, y_train, y_val, y_test = X
    
    LR = LogisticRegression(max_iter=300)
    
    if param == 'full':
        print("=========================================")
        print(f"\033[1m Logistic Regression Testing Results on Full Data\033[0m")
        print("=========================================")
        
        xtrain, xval, xtest= X_train, X_val, X_test
        best_param_dict = {'warm_start': False, 'solver': 'lbfgs', 'penalty': 'none', 'multi_class': 'multinomial',
                           'fit_intercept': True, 'dual': False, 'C': 0.7431578947368421}
        
    elif param == 'pca':
        
        print("=========================================")
        print(f"\033[1m Logistic Regression Testing Results on PCA Data\033[0m")
        print("=========================================")
        
        xtrain, xval, xtest = pd.DataFrame(X_train_pca), pd.DataFrame(X_val_pca), pd.DataFrame(X_test_pca)
        best_param_dict = {'warm_start': False, 'solver': 'lbfgs', 'penalty': 'none',
                           'multi_class': 'multinomial', 'fit_intercept': True, 'dual': False,
                           'C': 0.7431578947368421}
        
    start_LR_tune = time.time()


    
    best_model = LogisticRegression(max_iter=300, warm_start = best_param_dict['warm_start'],
                                    solver = best_param_dict['solver'], penalty = best_param_dict['penalty'],
                                    multi_class = best_param_dict['multi_class'], 
                                    fit_intercept = best_param_dict['fit_intercept'], 
                                    dual = best_param_dict['dual'], 
                                    C = best_param_dict['C']).fit(pd.concat([xtrain, xval]), pd.concat([y_train, y_val]))
    stop_LR_tune = time.time()
        
    time_to_complete = stop_LR_tune - start_LR_tune
    
    preds = best_model.predict(xtest)

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
    
    print('LOGISTIC REGRESSION BEST MODEL BASED ON TESTING DATA:')
    
    print(pt)
    
    print('The best parameters are: ', best_param_dict)
    
    return (preds, best_model)

