#!/usr/bin/env python
# coding: utf-8

# # Install required libraries

# In[1]:


# # uncomment and run this cell to install packages
# !pip3 install pandas scikit-learn xgboost workalendar==14.0.0
# # after installing packages restart kernel


# # Import required libraries

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from workalendar.asia import HongKong


# # Read test and train data sets from csv

# In[3]:


testData = pd.read_csv('test.csv')
trainData = pd.read_csv('train.csv')
trainData.head()


# # Feature engineering

# In[4]:


# date time formating
date_format_string ='%d/%m/%Y %H:%M'
testData.date = pd.to_datetime(testData.date, format=date_format_string)
trainData.date = pd.to_datetime(trainData.date, format=date_format_string)
# feature creation 
def FeatureCreation(Data):    
    Data['year']=Data['date'].dt.year 
    Data['month']=Data['date'].dt.month 
    Data['day']=Data['date'].dt.day
    Data['dayofweek_num']=Data['date'].dt.dayofweek
    Data['Hour'] = Data['date'].dt.hour
    Data['weekofyear'] = Data['date'].dt.weekofyear
    Data['date_only'] = Data['date'].dt.date
    return Data


# # Getting holiday dates for years 2017 and 2018

# In[5]:


cal = HongKong()
holidays_2017 = pd.DataFrame(cal.holidays(2017),columns =['date_only','holiday'])
holidays_2018 = pd.DataFrame(cal.holidays(2018),columns =['date_only','holiday'])
holidays = pd.concat([holidays_2017,holidays_2018],ignore_index=True, sort=False)
holidays = holidays.drop('holiday', axis=1)
holidays['is_holiday'] = 1
holidays.head()


# # Train, validation and test sets

# In[6]:


# feature creation 
trainData = FeatureCreation(trainData)
# adding is holiday feature, it is 1 if the day is holiday else 0
trainData =  pd.merge(trainData,holidays, how='left', on=['date_only'])
trainData['is_holiday'] = trainData['is_holiday'].fillna(0)
testData = FeatureCreation(testData)
testData =  pd.merge(testData,holidays, how='left', on=['date_only'])
testData['is_holiday'] = testData['is_holiday'].fillna(0)
# features
features = ['year','month','day','dayofweek_num','Hour','weekofyear','is_holiday']
X = trainData[features]
y = trainData.speed
X_test = testData[features]
# train test split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[7]:


print(train_X)


# # Model XGBoost Regression 

# In[8]:


# XGBRegressor model
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# fit model with train features and label. early stopping prevents overfitting
my_model.fit(train_X, train_y, 
             early_stopping_rounds=5, 
             eval_set=[(val_X, val_y)], 
             verbose=False)
# validation set MSE 
mse = mean_squared_error(my_model.predict(val_X), val_y)
print(mse)


# # Model fitting and prediction

# In[9]:


# fitting model with full data set
my_model.fit(X, y,
             early_stopping_rounds=5, 
             eval_set=[(val_X, val_y)], 
             verbose=False)


# In[10]:


# prediction for test data set 
preds = my_model.predict(X_test)
# storing predictions into csv file
df = pd.DataFrame(preds)
pred = pd.DataFrame()
pred['id'] = df.index
pred['speed']=df[0]
pred.head()
pred.to_csv('test_predictions.csv', index = False)

