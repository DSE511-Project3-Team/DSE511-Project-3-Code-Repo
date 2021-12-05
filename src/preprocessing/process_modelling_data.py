import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from src.preprocessing.preprocessing_functions import *

def get_modelling_data():
    # Supress settingwithcopywarning
    pd.set_option('mode.chained_assignment', None)

    # Construct the relative raw data file path
    dirname = os.path.dirname(__file__)
    relative_path = '../../data/processed/imputed.pkl'
    datafile = os.path.join(dirname, relative_path)

    data = ()
    if os.path.exists(datafile):
        X = pd.read_pickle(datafile)
        X_ = X.copy()
        X_train, X_val, X_test, y_train, y_val, y_test = encode_std_extract_split(X_)
        pca = PCA(n_components=0.95, svd_solver = 'full')
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_val_pca = pca.transform(X_val)
        X_test_pca = pca.transform(X_test)    
        data = (X_train, X_val, X_test, X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test)
    else:
        print("imputed data does not exisits")
    
    return data