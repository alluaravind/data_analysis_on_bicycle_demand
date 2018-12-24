#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 10:58:00 2018

"""

#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor



df_train = pd.read_csv('train.csv')
df_test= pd.read_csv('test.csv')


X_train = df_train[['season','holiday','workingday','weather','temp','atemp','humidity','windspeed']]
y_train= df_train['count']
X_test_data= df_test[['season','holiday','workingday','weather','temp','atemp','humidity','windspeed']]

print(X_train.shape)

forest = RandomForestRegressor(n_estimators=200,random_state=0)
clf = forest.fit(X_train, y_train)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", align="center")
plt.show()


