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



def scale_data(train_X, validate_X, test_X):
    # Scale the data
    scaler = sklearn.preprocessing.MinMaxScaler()

    # Fit the scaler
    scaler.fit(train_X)

    # Use the scaler to transform train, validate, test
    X_train_scaled = scaler.transform(train_X)
    X_validate_scaled = scaler.transform(validate_X)
    X_test_scaled = scaler.transform(test_X)


    # Turn everything into a dataframe
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=train_X.columns)
    X_validate_scaled = pd.DataFrame(X_validate_scaled, columns=train_X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=train_X.columns)
    return X_train_scaled, X_validate_scaled, X_test_scaled




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
    
    
    
    
def make_metric_df(y, y_pred, model_name, metric_df):
    if metric_df.size ==0:
        metric_df = pd.DataFrame(data=[
            {
                'model': model_name, 
                'RMSE_validate': mean_squared_error(
                    y,
                    y_pred) ** .5,
                'r^2_validate': explained_variance_score(
                    y,
                    y_pred)
            }])
        return metric_df
    else:
        return metric_df.append(
            {
                'model': model_name, 
                'RMSE_validate': mean_squared_error(
                    y,
                    y_pred) ** .5,
                'r^2_validate': explained_variance_score(
                    y,
                    y_pred)
            }, ignore_index=True)
    
    
    
    
def linear_regression(train_X, train_y, validate_X, validate_y):
    lm = LinearRegression(normalize=True)
    lm.fit(train_X, train_y.log_error)
    train_y['log_error_pred_lm'] = lm.predict(train_X)
    
    # evaluate: rmse
    rmse_train = mean_squared_error(train_y.log_error, train_y.log_error_pred_lm) ** (1/2)

    # predict validate
    validate_y['log_error_pred_lm'] = lm.predict(validate_X)

    # evaluate: rmse
    rmse_validate = mean_squared_error(validate_y.log_error, validate_y.log_error_pred_lm) ** (1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
    return rmse_train, rmse_validate




def lassolars(train_X, train_y, validate_X, validate_y):
    # create the model object
    lars = LassoLars(alpha=1)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series!
    lars.fit(train_X, train_y.log_error)

    # predict train
    train_y['log_error_pred_lars'] = lars.predict(train_X)

    # evaluate: rmse
    rmse_train = mean_squared_error(train_y.log_error, train_y.log_error_pred_lars) ** (1/2)

    # predict validate
    validate_y['log_error_pred_lars'] = lars.predict(validate_X)

    # evaluate: rmse
    rmse_validate = mean_squared_error(validate_y.log_error, validate_y.log_error_pred_lars) ** (1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
    return rmse_train, rmse_validate




def polynomial(train_X, train_y, validate_X, validate_y, test_X):
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(train_X)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(validate_X)
    X_test_degree2 =  pf.transform(test_X)
    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, train_y.log_error)

    # predict train
    train_y['log_error_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(train_y.log_error, train_y.log_error_pred_lm2) ** (1/2)

    # predict validate
    validate_y['log_error_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(validate_y.log_error, validate_y.log_error_pred_lm2) ** 0.5

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
    return rmse_train, rmse_validate




