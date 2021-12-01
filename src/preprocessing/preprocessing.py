#!/usr/bin/env python
# coding: utf-8

# In[1]:


from src.preprocessing.preprocessing_functions import *


# In[2]:


relative_path = 'C:\\Users\\Russ\\Desktop\\DSE511-Project-3-Code-Repo\\data\\raw\\accident_data.csv'
accident_data = pd.read_csv(relative_path)

city_list = ['Phoenix', 'Los Angeles', 'New York', 'Philadelphia', 'Houston', 'Chicago']
state_list = ['AZ', 'CA', 'NY', 'PA', 'TX', 'IL']


# In[3]:


accident_data = isolate_city_state(accident_data, city_list, state_list)


# In[4]:


temp_wind = subset_df(accident_data, ['Temperature(F)', 'Wind_Speed(mph)'])
OLS(temp_wind, np.array(accident_data['Wind_Chill(F)']))

accident_data["Wind_Temp"] = accident_data['Wind_Chill(F)'].fillna((accident_data['Temperature(F)']*1.0178 - accident_data['Wind_Speed(mph)']*0.3023))


# In[5]:


non_essential_features = ['ID', 'Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng', 'End_Lat',
        'End_Lng', 'Description', 'Number', 'Street', 'State', 'Zipcode', 'Country',
        'Airport_Code', 'Weather_Timestamp', 'Roundabout', 'Civil_Twilight', 'Nautical_Twilight',
        'Astronomical_Twilight', 'Wind_Chill(F)']

accident_data.drop(non_essential_features, axis=1, inplace=True)


# In[6]:


print('Percent of missing rows by column', '\n\n', accident_data.isnull().sum()/len(accident_data))


# In[8]:


# This takes some time to run:

env_vars_numeric = ['Temperature(F)', 'Wind_Temp', 'Humidity(%)', 'Pressure(in)', 
                    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']


Imputed_env_vars = knn_imputer(subset_df(accident_data, env_vars_numeric), k=2)


# In[9]:


Imputed_env_vars=basic_impute(Imputed_env_vars)
print(Imputed_env_vars.isnull().sum()/len(Imputed_env_vars))
accident_data[env_vars_numeric] = Imputed_env_vars
print(accident_data)

