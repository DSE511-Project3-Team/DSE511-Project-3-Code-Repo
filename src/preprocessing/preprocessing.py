#!/usr/bin/env python
# coding: utf-8

from src.preprocessing.preprocessing_functions import *


relative_path = 'C:\\Users\\russ_\\OneDrive\\Desktop\\US_Accidents_Dec20_updated.csv'
accident_data = pd.read_csv(relative_path)

city_list = ['Phoenix', 'Los Angeles', 'New York', 'Philadelphia', 'Houston', 'Chicago']
state_list = ['AZ', 'CA', 'NY', 'PA', 'TX', 'IL']

accident_data = isolate_city_state(accident_data, city_list, state_list)

temp_wind = subset_df(accident_data, ['Temperature(F)', 'Wind_Speed(mph)'])
OLS(temp_wind, np.array(accident_data['Wind_Chill(F)']))

accident_data["Wind_Temp"] = accident_data['Wind_Chill(F)'].fillna((accident_data['Temperature(F)']*1.0178 - accident_data['Wind_Speed(mph)']*0.3023))

Severity = accident_data.Severity

non_essential_features = ['ID', 'Start_Time', 'Severity', 'End_Time', 'Start_Lat', 'Start_Lng', 'End_Lat',
        'End_Lng', 'Description', 'Number', 'Street', 'State', 'Zipcode', 'Country',
        'Airport_Code', 'Weather_Timestamp', 'Roundabout', 'Civil_Twilight', 'Nautical_Twilight',
        'Astronomical_Twilight', 'Wind_Chill(F)']

accident_data.drop(non_essential_features, axis=1, inplace=True)

print('Percent of missing rows by column', '\n\n', accident_data.isnull().sum()/len(accident_data))

# This takes some time to run:

env_vars_numeric = ['Temperature(F)', 'Wind_Temp', 'Humidity(%)', 'Pressure(in)', 
                    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

Imputed_env_vars = knn_imputer(subset_df(accident_data, env_vars_numeric), k=2)

Imputed_env_vars=basic_impute(Imputed_env_vars)
print(Imputed_env_vars.isnull().sum()/len(Imputed_env_vars))
accident_data[env_vars_numeric] = Imputed_env_vars
accident_data['Severity'] = Severity
