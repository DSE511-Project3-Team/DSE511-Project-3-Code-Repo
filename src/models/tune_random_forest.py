from sklearn.ensemble import RandomForestClassifier

import copy 
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score 
from prettytable import PrettyTable
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def perform_random_forest_tuning(X, param):
    X_train, X_val, X_test, X_train_pca, X_val_pca, \
                X_test_pca, y_train, y_val, _ = X
    
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 100)]
    max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    criterion = ['gini', 'entropy']
    max_features = ['auto', 'sqrt', 'log2']
    
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'criterion': criterion,
                   'max_features': max_features}
    
    RF = RandomForestClassifier(random_state = 44)
    
    if param == 'full':
        print("=========================================")
        print(f"\033[1m Random Forest Tuning Results on Full Data\033[0m")
        print("=========================================")
        
        xtrain, xval, xtest= X_train, X_val, X_test
        
    elif param == 'pca':
        
        print("=========================================")
        print(f"\033[1m Random Forest Tuning Results on PCA Data\033[0m")
        print("=========================================")
        
        xtrain, xval, xtest = pd.DataFrame(X_train_pca), pd.DataFrame(X_val_pca), pd.DataFrame(X_test_pca)
        
    start = time.time()

    RF_random = RandomizedSearchCV(estimator = RF,
                                    param_distributions = random_grid, 
                                    n_iter = 100, cv = 3, 
                                    verbose= 2, random_state=44, refit = callable,
                                    n_jobs = -1, scoring = ['f1_macro', 'accuracy']) 
        

    RF_random.fit(pd.concat([xtrain, xval]), pd.concat([y_train, y_val]))

    stop = time.time()
        
    time_to_complete = stop - start
    best_param_dict = RF_random.best_params_
    
    best_model = RandomForestClassifier(random_state = 44,
                                n_estimators = best_param_dict['n_estimators'],
                                max_depth = best_param_dict['max_depth'],
                                min_samples_split = best_param_dict['min_samples_split'], 
                                min_samples_leaf = best_param_dict['min_samples_leaf'],
                                criterion = best_param_dict['criterion'],
                                max_features = best_param_dict['max_features']).fit(pd.concat([xtrain, xval]), pd.concat([y_train, y_val]))
    
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
    
    print('RANDOM FOREST BEST MODEL BASED ON VALIDATION:')
    
    print(pt)
    
    print('The best parameters are: ', best_param_dict)
    
    return (preds, best_model)
    