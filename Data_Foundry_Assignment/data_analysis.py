#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk


# In[36]:


df_train = pd.read_csv('train.csv')
df_test= pd.read_csv('test.csv')


# In[37]:


X_train = df_train[['season','holiday','workingday','weather','temp','atemp','humidity','windspeed']]
y_train= df_train['count']
X_test_data= df_test[['season','holiday','workingday','weather','temp','atemp','humidity','windspeed']]


# In[39]:


from sklearn.ensemble import RandomForestRegressor

regr_rf = RandomForestRegressor(n_estimators=200,max_depth=3,random_state=0)
clf = regr_rf.fit(X_train, y_train)


# In[40]:


df_test['predicted_count'] = [round(data) for data in regr_rf.predict(X_test_data)]


# In[41]:


writer = pd.ExcelWriter('C:/Users/cnadikot/Desktop/predicted_data.xlsx')
df_test.to_excel(writer,'Sheet1')
writer.save()


# In[42]:


import pickle;
# open a file, where you ant to store the data
file = open('important', 'wb')

# dump information to that file
pickle.dump(clf, file)

# close the file
file.close()

# open a file, where you stored the pickled data
file = open('important', 'rb')

# dump information to that file
clf2 = pickle.load(file)

# close the file
file.close()


# In[45]:


#we can also use clf.predict for predicting data after loading model into pickel file
clf.predict(X_test_data)


# In[ ]:




