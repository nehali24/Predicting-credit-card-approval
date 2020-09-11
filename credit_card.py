#!/usr/bin/env python
# coding: utf-8

# This is a data set thet ahas been aquired from kaggle. As commercial banks receive alot of applications for credit cards and many of them get rejected due to various reasons. Here in this project i have tried using machine learning to predict the approval of credit cards.

# In[21]:


import pandas as pd
cc_apps = pd.read_csv("cc_approvals.data")


# While exploring the data i got some mixed results that was quite unclear.this dataset has numerical as well as non numerical data.

# In[22]:


cc_apps.head(5)


# In[23]:


cc_apps_description = cc_apps.describe()
print(cc_apps_description)


# In[24]:


cc_apps_info = cc_apps.info()
print(cc_apps_info)


# Feature number 2, 7, 10, 14 has float or int type, and rest all features are non numeric.

# In[25]:


cc_apps.tail(17)


# Replacing ? with NaN

# In[26]:


import numpy as np

cc_apps.replace('?', 'NaN')


# Imputing missing values to avoid any problem in our model.

# In[27]:


cc_apps.fillna(cc_apps.mean(), inplace = True)
print(cc_apps.isnull().sum())


# Imputing non numeric data. 

# In[28]:


for col in cc_apps.columns:
    if cc_apps[col].dtypes == 'object':
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])
print(cc_apps.isnull().sum())


# Preprocessing data: Coverting non numeric to numeric.

# In[29]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cc_apps.columns:
    if cc_apps[col].dtypes == 'object':
        cc_apps[col] = le.fit_transform(cc_apps[col])


# In[30]:


cc_apps


# Splitting the data into training and test set, dropping feaures like DriversLicence and ZipCode as they are not relevant.

# In[31]:


from sklearn.model_selection import train_test_split
cc_apps = cc_apps.drop([11,13])
cc_apps = cc_apps.values
X,y = cc_apps[:,0:13], cc_apps[:,13]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.33)


# Fitting the model.

# In[32]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)


# Logistic Regression Model Fitting to our training set.

# In[33]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(rescaledX_train, y_train)


# In[34]:



from sklearn.metrics import confusion_matrix


y_pred = logreg.predict(rescaledX_test)


print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test,y_test))


confusion_matrix(y_test,y_pred)


# From our confusion matrix we can see that accuracy is pretty low.

# In[35]:


from sklearn.model_selection import GridSearchCV
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]
param_grid = {'tol':tol, 'max_iter': max_iter}


# In[38]:


grid_model = GridSearchCV(logreg,param_grid,cv = 5)
rescaledX = scaler.fit_transform(X)
grid_model_result = grid_model.fit(rescaledX,y)
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print(best_score, best_params)


# Best Score : 0.186
