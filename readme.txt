Programming Languages: Phyton

Required Library Packages : pandas, scikit-learn, xgboost, workalendar(version 14)
pandas : using pandas to read data from csv and store into DataFrame and to perform feature engneering 

scikit-learn : 
from scikit-learn sklearn.model_selection import train_test_split
->using function train_test_split to split train dataset into train and validation sets default ratio 3:1
from sklearn.metrics import mean_squared_error
->using function mean_squared_error to measure the mean squared error of predicted label of validation set

xgboost:
from xgboost import XGBRegressor
->using XGBRegressor model for training and prediction

workalendar(version 14):
from workalendar.asia import HongKong
->using workalendar.asia HongKong() to get holiday dates in HongKong for year 2017 and 2018

Running jupyter notebook:
1. installing packages : install packages in jupyter notebook using below command(restart kernel after installing the packages)
!pip3 install pandas scikit-learn xgboost workalendar==14.0.0
2. Run all cells in notebook 

 