import copy 
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score 
from prettytable import PrettyTable
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler


def perform_multinomial_naive_bayes_tuning(X, param):
    
    X_train, X_val, X_test, X_train_pca, X_val_pca, \
                X_test_pca, y_train, y_val, _ = X
        
    
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    scaler = MinMaxScaler()
    scaler.fit(X_val)
    X_val = scaler.transform(X_val)

    scaler = MinMaxScaler()
    scaler.fit(X_test)
    X_test=scaler.transform(X_test)

    scaler = MinMaxScaler()
    scaler.fit(X_train_pca)
    X_train_pca=scaler.transform(X_train_pca)

    scaler = MinMaxScaler()
    scaler.fit(X_val_pca)
    X_val_pca=scaler.transform(X_val_pca)

    scaler = MinMaxScaler()
    scaler.fit(X_test_pca)
    X_test_pca=scaler.transform(X_test_pca)
    
    
    alpha = np.linspace(0.001, 0.3, 3000)
    
    tuned_parameters = {'fit_prior': [True, False],
                        'alpha': alpha}
    
    clf = MultinomialNB()
    
    if param == 'full':
        print("=========================================")
        print(f"\033[1m Multinomial Naive Bayes Tuning Results on Full Data\033[0m")
        print("=========================================")
        
        xtrain, xval, xtest= pd.DataFrame(X_train), pd.DataFrame(X_val), pd.DataFrame(X_test)
        
    elif param == 'pca':
        
        print("=========================================")
        print(f"\033[1m Multinomial Naive Bayes Tuning Results on PCA Data\033[0m")
        print("=========================================")
        
        xtrain, xval, xtest = pd.DataFrame(X_train_pca), pd.DataFrame(X_val_pca), pd.DataFrame(X_test_pca)
        
    start = time.time()

    clf_random = RandomizedSearchCV(clf, 
                                    tuned_parameters, n_iter = 2000,
                                    cv=8, 
                                    verbose= 2, random_state=44,
                                    n_jobs = -1,
                                    scoring='f1_macro')
    
    clf_random.fit(pd.concat([xtrain, xval]), pd.concat([y_train, y_val]))

    stop = time.time()
        
    time_to_complete = stop - start
    best_param_dict = clf_random.best_params_
    
    
    best_model = MultinomialNB(fit_prior = best_param_dict['fit_prior'],
                               alpha = best_param_dict['alpha']).fit(pd.concat([xtrain, xval]), pd.concat([y_train, y_val]))

    
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
    
    print('MULTINOMIAL BAYES BEST MODEL BASED ON VALIDATION:')
    
    print(pt)
    
    print('The best parameters are: ', best_param_dict)
    
    return (preds, best_model)
