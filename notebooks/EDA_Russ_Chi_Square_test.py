#!/usr/bin/env python
# coding: utf-8

# ## Chi Squared Test for Homoeneity

# In[1]:


import copy 
import sys

path_to_dataset = 'C:\\Users\\russ_\\OneDrive\\Desktop\\US_Accidents_Dec20_updated.csv'
sys.path.append(path_to_dataset)
sys.path.append('C:\\Users\\russ_\\OneDrive\\Desktop\\Newest_repos\\DSE511-Project-3-Code-Repo')


# In[3]:


from src.preprocessing.preprocessing import *


# In[15]:


from scipy.stats import chisquare

accident_data_chisquared = copy.copy(accident_data)
accident_data_chisquared['Severity'] = (accident_data_chisquared['Severity'] < 3).astype(int)  

print(accident_data_chisquared[['Severity', 'City']].groupby(['City']).sum())

severe_accidents = np.array(accident_data_chisquared[['Severity', 'City']].groupby(['City']).sum()).reshape(1, -1)
freq = np.array(accident_data_chisquared[['Severity', 'City']].groupby(['City']).count()).reshape(1, -1)

observered1 = copy.copy(severe_accidents)
observered0 = freq - severe_accidents

expected1 = (observered1.sum()*freq)/len(accident_data_chisquared)
expected0 = (observered0.sum()*freq)/len(accident_data_chisquared)

ChiSqrResults = chisquare(f_obs = observered1.T, f_exp = expected1.T)
print('\n\n','Our Chi-Square Results produce a p-value of ', ChiSqrResults.pvalue[0],
      'with a test statistic of ', ChiSqrResults.statistic[0], 
      ' therefore, we can conclude statistically, that the ratio of severe accidents to total accidents varies by city.')


# In[20]:


import matplotlib.pyplot as plt

ratio_of_severe_accidents = observered1/freq

plt.figure(figsize = (10, 5))
cities = list(accident_data_chisquared['City'].unique())

plt.bar(cities, ratio_of_severe_accidents[0].tolist(), color = 'maroon')
 
plt.xlabel("Largest Cities in the US")
plt.ylabel("# Serious to Total Accidents")
plt.title("Ratio of Accident Severity by Most Populated US Cities")
plt.show()


# In[ ]:




