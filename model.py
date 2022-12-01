import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

# acquire
from env import get_db_url
from pydataset import data
import seaborn as sns

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")



def model_prep(train, validate, test):
    '''Prepare train, validate, and test data for modeling'''
    
    # drop unused columns
    keep_cols = ['log_error',
                 'Orange',
                 'LA',
                 'Ventura',
                 'age',
                 'age_bin',
                 'taxrate',
                 'acres',
                 'acres_bin',
                 'sqft_bin',
                 'latitude',
                 'longitude',
                 'calc_sqft',
                 'structure_dollar_per_sqft',
                 'structure_dollar_sqft_bin',
                 'land_dollar_per_sqft',
                 'lot_dollar_sqft_bin',
                 'bath_bed_ratio',
                 'cola'
                ]
    
    train = train[keep_cols]
    validate = validate[keep_cols]
    test = test[keep_cols]
    
    # Split data into predicting variables (X) and target variable (y) and reset the index for each dataframe
    train_X = train.drop(columns='log_error').reset_index(drop=True)
    train_y = train[['log_error']].reset_index(drop=True)
    
    validate_X = validate.drop(columns='log_error').reset_index(drop=True)
    validate_y = validate[['log_error']].reset_index(drop=True)
    
    test_X = test.drop(columns='log_error').reset_index(drop=True)
    test_y = test[['log_error']].reset_index(drop=True)
    
    return train_X, validate_X, test_X, train_y, validate_y, test_y




def get_mean_median(train_y, validate_y):
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values.
    y_train = pd.DataFrame(train_y)
    y_validate = pd.DataFrame(validate_y)
    
    # Predict property_value_pred_mean
    log_error_pred_mean = train_y.log_error.mean()
    train_y['log_error_pred_mean'] = log_error_pred_mean
    validate_y['log_error_pred_mean'] = log_error_pred_mean
    
    # compute property_value_pred_median
    log_error_pred_median = train_y.log_error.median()
    train_y['log_error_pred_median'] = log_error_pred_median
    validate_y['log_error_pred_median'] = log_error_pred_median
    
    # RMSE of property_value_pred_mean
    rmse_train = mean_squared_error(y_train.log_error,
                                y_train.log_error_pred_mean) ** .5
    rmse_validate = mean_squared_error(y_validate.log_error, y_validate.log_error_pred_mean) ** (1/2)
    
    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 3), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 3))
    
    # RMSE of property_value_pred_median
    rmse_train = mean_squared_error(y_train.log_error, y_train.log_error_pred_median) ** .5
    rmse_validate = mean_squared_error(y_validate.log_error, y_validate.log_error_pred_median) ** .5
    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 3), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 3))