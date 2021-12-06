import time
import numpy as np
import pandas as pd
import sys
import os

from src.preprocessing.preprocessing_functions import *

def perform_imputation(six_cities_df):
    six_cities_df["Wind_Temp"] = six_cities_df['Wind_Chill(F)'].fillna((six_cities_df['Temperature(F)']*1.0178 - six_cities_df['Wind_Speed(mph)']*0.3023))
    six_cities_df = six_cities_df.drop("Wind_Chill(F)", axis=1)
    env_vars_numeric = ['Temperature(F)', 'Wind_Temp', 'Humidity(%)', 'Pressure(in)', 
                        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

    # Perform KNN Imputation
    print("Performing knn imputation ...")
    start_time = time.time()
    Imputed_env_vars = knn_imputer(subset_df(six_cities_df, env_vars_numeric), k=3)
    knn_imputation_time = time.time() - start_time
    print(f"KNN Imputation Time {knn_imputation_time / 60} mins")

    # Perform Basic Imputation
    print("Performing basic imputation")
    Imputed_env_vars=basic_impute(Imputed_env_vars)
    six_cities_df[env_vars_numeric] = Imputed_env_vars  

    ## Impute Categorical Variables
    print("Perform imputation on categorical variables")
    six_cities_df['Wind_Direction'] = six_cities_df['Wind_Direction'].fillna(six_cities_df['Wind_Direction'].mode()[0])
    six_cities_df['Weather_Condition'] = six_cities_df['Weather_Condition'].fillna(six_cities_df['Weather_Condition'].mode()[0])
    six_cities_df.isna().sum() / six_cities_df.shape[0] * 100  

    return six_cities_df


def generate_imputed_data():
    
    # Construct the relative raw data file path
    dirname = os.path.dirname(__file__)
    relative_path_raw = '../../data/raw/accident_data.csv'
    datafile_raw = os.path.join(dirname, relative_path_raw)

    # Construct the relative processed data file path
    dirname = os.path.dirname(__file__)
    relative_path_pro = '../../data/processed/imputed.pkl'
    datafile_pro = os.path.join(dirname, relative_path_pro)

    if os.path.exists(datafile_raw):
        # Delete old imputed file
        if os.path.exists(datafile_pro):
            os.remove(datafile_pro)
            print("Deleted old imputed file")

        df = pd.read_csv(datafile_raw)
        six_cities_df = df.copy()
        six_cities_df = perform_imputation(six_cities_df)
        six_cities_df = six_cities_df.reset_index(drop=True)
        pd.to_pickle(six_cities_df, datafile_pro)
        print("saved the imputed file")
    else:
        print("raw file path does not exist")
        sys.exit()

    print("Job completed, check the imputed file in data/processed folder.")