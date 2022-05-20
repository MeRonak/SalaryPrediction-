#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


# In[3]:


df = pd.read_csv('D:\Downloads\Salary_Data.csv')


# In[28]:


df.head()
df.isna().sum()
#No null value of empty value 


# In[7]:


x = np.array(df["YearsExperience"])
y = np.array(df["Salary"])


# In[21]:


plt.xlabel("Years of Experience")
plt.ylabel("salary in rupee")
plt.scatter(x,y,color = 'red', marker = '+')


# In[34]:


x = x.reshape(-1,1)
y = y.reshape(-1,1)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)
pred = lr.predict(x)
plt.scatter(x,y)
plt.plot(x,pred)


# In[37]:


x = np.array([1])
x = x.reshape(-1,1)
lr.predict(x)


# In[ ]:




