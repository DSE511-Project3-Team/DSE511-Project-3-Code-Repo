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


def perform_Logistic_Regression_tuning(X, param):
    X_train, X_val, X_test, X_train_pca, X_val_pca, \
                X_test_pca, y_train, y_val, _ = X
    
    grid = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'dual': [False],
            'fit_intercept': [True, False],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'C': np.linspace(0.01, 2, 20),
            'multi_class': ['auto', 'ovr', 'multinomial'],
            'warm_start': [True, False]}

    
    LR = LogisticRegression(max_iter=300)
    
    if param == 'full':
        print("=========================================")
        print(f"\033[1m Logistic Regression Tuning Results on Full Data\033[0m")
        print("=========================================")
        
        xtrain, xval, xtest= X_train, X_val, X_test
        
    elif param == 'pca':
        
        print("=========================================")
        print(f"\033[1m Logistic Regression Tuning Results on PCA Data\033[0m")
        print("=========================================")
        
        xtrain, xval, xtest = pd.DataFrame(X_train_pca), pd.DataFrame(X_val_pca), pd.DataFrame(X_test_pca)
        
    start_LR_tune = time.time()

    LR_random = RandomizedSearchCV(estimator = LR,
                                    param_distributions = grid, 
                                    n_iter = 75, cv = 5, 
                                    verbose= 0, random_state=44, refit = callable,
                                    n_jobs = -1, scoring = ['f1_macro', 'accuracy'])
        
    # I combined the training and validation data because the RandomizedSearchCV
    # uses cross validation to determine optimal parameters. 

    LR_random.fit(pd.concat([xtrain, xval]), pd.concat([y_train, y_val]))

    stop_LR_tune = time.time()
        
    time_to_complete = stop_LR_tune - start_LR_tune
    best_param_dict = LR_random.best_params_
    
    best_model = LogisticRegression(max_iter=300, warm_start = best_param_dict['warm_start'],
                                    solver = best_param_dict['solver'], penalty = best_param_dict['penalty'],
                                    multi_class = best_param_dict['multi_class'], 
                                    fit_intercept = best_param_dict['fit_intercept'], 
                                    dual = best_param_dict['dual'], 
                                    C = best_param_dict['C']).fit(pd.concat([xtrain, xval]), pd.concat([y_train, y_val]))

    best_model.fit(pd.concat([xtrain, xval]), pd.concat([y_train, y_val]))

    preds = best_model.predict(pd.concat([xtrain, xval]))

    conf_mat = confusion_matrix(pd.concat([y_train, y_val]), preds)
    accuracy = accuracy_score(pd.concat([y_train, y_val]), preds)
    

    pt = PrettyTable(['Time to Tune (s)', 'Accuracy', 'Sensitivity', 
                          'Specificity', 'Precision', 'F1 Score (macro)']) 

    pt.add_row([round(time_to_complete, 2), round(accuracy, 2), 
                round(conf_mat[1,1]/(conf_mat[1,1]+conf_mat[1,0]), 2),
                round(conf_mat[0,0]/(conf_mat[0,1]+conf_mat[0,0]), 2),
                round(conf_mat[1,1]/(conf_mat[1,1]+conf_mat[0,1]), 2),
                round(f1_score(pd.concat([y_train, y_val]), preds, average = 'macro'), 2)])
    
    conf_mat1 = PrettyTable(['Confusion Matrix','Predicted Pos.', 'Predicted Neg.'])
    conf_mat1.add_row(['Actual Postive', conf_mat[1,1], conf_mat[1,0]])
    conf_mat1.add_row(['Actual Negative', conf_mat[0,1], conf_mat[0,0]])

    print(conf_mat1)
    
    
    print('LOGISTIC REGRESSION BEST MODEL BASED ON VALIDATION:')
    display(pt)
    print('The best parameters are: ', best_param_dict)
    
    return (preds, best_model)
