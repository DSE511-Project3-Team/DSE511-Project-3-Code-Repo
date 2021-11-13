#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import statsmodels.api as sm
from numpy import NaN

dirname = os.path.dirname(__file__)
relative_path = '../../data/raw/accident_data.csv'
datafile = os.path.join(dirname, relative_path)

raw_data = pd.read_csv(datafile)
print(raw_data.shape)

city_list = ['Phoenix', 'Los Angeles', 'New York', 'Philadelphia', 'Houston', 'Chicago']
state_list = ['AZ', 'CA', 'NY', 'PA', 'TX', 'IL']

def isolate_city_state(data, cities, states):
  
    ''' This ensures that each city is selected with it's respective state
    which is why I didn't simply run a merge statement.'''
    
    for x in zip(cities, states):
        df_x = data.loc[(data['City'] == x[0]) & (data['State'] == x[1])]
        df_x.append(df_x)
    
    return df_x


raw_data = isolate_city_state(raw_data, city_list, state_list)
print(raw_data.shape, '\n\n', raw_data.head())

env_vars = ['Weather_Timestamp', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 
                    'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition', 
                    'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']

def subset_df(df, keep_list):
    mask = df.columns.isin(keep_list)
    selectedCols = df.columns[mask]
    return df[selectedCols]

raw_data = subset_df(raw_data, env_vars)

print('Percent of missing rows by column', '\n\n', raw_data.isnull().sum()/len(raw_data))

def OLS(x, y):    
    model = sm.OLS(y, x, missing='drop')
    results = model.fit()
    print(results.summary())

def basic_impute(data):
    
    df_num = data.select_dtypes(include=np.number)

    for i in data:
        if i in df_num:
            data.loc[data.loc[:,i].isnull(),i] = df_num.loc[:,i].median()
        else:
            data.loc[data.loc[:,i].isnull(),i] = data.loc[:,i].mode()
    
    return data

x = subset_df(raw_data, ['Temperature(F)', 'Wind_Speed(mph)'])
OLS(x, np.array(raw_data['Wind_Chill(F)']))

raw_data['Wind_Chill(F)'].fillna((raw_data['Temperature(F)']*1.0778 + raw_data['Wind_Speed(mph)']*-0.7083), inplace=True)

print('Percent of missing rows by column', '\n\n', raw_data.isnull().sum()/len(raw_data))

raw_data = basic_impute(raw_data)

print('Percent of missing rows by column', '\n\n', raw_data.isnull().sum()/len(raw_data))
