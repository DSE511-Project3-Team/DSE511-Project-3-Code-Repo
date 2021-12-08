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


def perform_gradient_boosting_tuning(X, param):
    
    X_train, X_val, X_test, X_train_pca, X_val_pca, \
                X_test_pca, y_train, y_val, _ = X
    
    loss = ['deviance', 'exponential']
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 300, num = 100)]
    learning_rate = np.linspace(0.001, 2, 500)
    max_depth = [int(x) for x in np.linspace(4, 50, 25)]
    max_features = ['auto', 'sqrt', 'log2']
    criterion = ['friedman_mse', 'squared_error', 'mse', 'mae']
    
    
    tuned_parameters = {'loss': loss,
                        'n_estimators': n_estimators,
                        'learning_rate': learning_rate,
                        'max_depth': max_depth,
                        'max_features': max_features,
                        'criterion': criterion}
    
    clf = GradientBoostingClassifier(random_state=44)
    
    if param == 'full':
        print("=========================================")
        print(f"\033[1m Gradient Boosting Tuning Results on Full Data\033[0m")
        print("=========================================")
        
        xtrain, xval, xtest= pd.DataFrame(X_train), pd.DataFrame(X_val), pd.DataFrame(X_test)
        
    elif param == 'pca':
        
        print("=========================================")
        print(f"\033[1m Gradient Boosting Tuning Results on PCA Data\033[0m")
        print("=========================================")
        
        xtrain, xval, xtest = pd.DataFrame(X_train_pca), pd.DataFrame(X_val_pca), pd.DataFrame(X_test_pca)
        
    start = time.time()

    clf_random = RandomizedSearchCV(clf, 
                                    tuned_parameters, n_iter = 5,
                                    cv=2, 
                                    verbose= 0, 
                                    random_state=44,
                                    n_jobs = -1,
                                    scoring='f1_macro')
    
    clf_random.fit(pd.concat([xtrain, xval]), pd.concat([y_train, y_val]))

    stop = time.time()
        
    time_to_complete = stop - start
    best_param_dict = clf_random.best_params_
    
    
    best_model = GradientBoostingClassifier(loss = best_param_dict['loss'],
                                           n_estimators = best_param_dict['n_estimators'],
                                           learning_rate = best_param_dict['learning_rate'],
                                           max_depth = best_param_dict['max_depth'],
                                           max_features = best_param_dict['max_features'],
                                           criterion = best_param_dict['criterion']).fit(pd.concat([xtrain, xval]), pd.concat([y_train, y_val]))

    
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
    
    print('GRADIENT BOOSTING BEST MODEL BASED ON VALIDATION:')
    
    print(pt)
    
    print('The best parameters are: ', best_param_dict)
    
    return (preds, best_model)
