# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 17:58:43 2018

@author: techwiz
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing,model_selection,ensemble,metrics

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

"""Data Wrangling """
#Encoding Categorial Variables
train = pd.DataFrame(train)
count=0
for i in train.columns:
    if(train[i].dtype == 'object'):
        lb = preprocessing.LabelEncoder()
        #imputing missing data
        train.iloc[:,count] = train.iloc[:,count].fillna('N')
        test.iloc[:,count] = test.iloc[:,count].fillna('N')
        train.iloc[:,count] = lb.fit_transform(train.iloc[:,count])
        test.iloc[:,count] = lb.fit_transform(test.iloc[:,count])
    count += 1
#imputing other missing data
train.fillna(method='ffill',inplace=True)
test.fillna(method='ffill',inplace=True)

train.drop(['Id','Street','Utilities'],axis=1,inplace=True)
test.drop(['Id','Street','Utilities'],axis=1,inplace=True)

""" Feature Selection and scaling """
X = train.iloc[:,0:77].values
y = train.iloc[:,[77]].values

scalar_X = preprocessing.MinMaxScaler(feature_range=(0.1,0.9))
scaler_y = preprocessing.StandardScaler()

X = scalar_X.fit_transform(X)
test = scalar_X.transform(test)
y = scaler_y.fit_transform(y)

"""Validating Model for best parameters"""
X_train , X_test , y_train , y_test = model_selection.train_test_split(X,y,test_size=0.2,random_state=42)


regressor = ensemble.RandomForestRegressor(n_estimators=500,warm_start=True,max_features="log2",max_depth=25)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

""" Metrics / Performance Evaluation """

y_true = scaler_y.inverse_transform(y_test)
y_pred_ = scaler_y.inverse_transform(y_pred)
rms_ = metrics.mean_squared_error(y_test,y_pred)
rms = metrics.mean_squared_error(y_true,y_pred_)

""" Traning Model on total training set """
regressor_ = ensemble.RandomForestRegressor(n_estimators=500,warm_start=True,max_features="log2",max_depth=25)
regressor_.fit(X,y)

""" predicting target"""
out = regressor_.predict(test)
out_ = scaler_y.inverse_transform(out)