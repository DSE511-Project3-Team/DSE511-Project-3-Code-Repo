#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import pickle


# In[21]:


data = pd.read_csv('C:\\...\\csv')


# In[23]:


print(data.shape)


# In[26]:


len(data[data.City == 'New York'])


# In[27]:


city_list = ['Phoenix', 'Los Angeles', 'New York', 'Philadelphia', 'Houston', 'Chicago']

data_cities = data[data['City'].isin(city_list)]
print(data_cities.shape)


# In[31]:


data_cities.to_csv('C:\\...\\csv')


# In[29]:


# Optional Pickle

#pickle_out = open("data_cities_pickle","wb")
#pickle.dump(data_cities, pickle_out)
#pickle_out.close()


# In[ ]:


#pickle_in = open("dict.pickle","rb")
#example_dict = pickle.load(pickle_in)

