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
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans

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



def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['latitude', 'longitude', 'age'],return_scaler=False):
    '''This function takes in train, validate, test, and outputs scaled data based on
    the chosen method (quantile scaling) using the columns selected as the only columns
    that will be scaled. This function also returns the scaler object as an array if set 
    to true'''
    # make copies of our original data
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
     # select a scaler
    scaler = MinMaxScaler()
     # fit on train
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled




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


def make_clusters(train,validate,test):
    #scale the data
    train_scaled, validate_scaled, test_scaled= scale_data(train,validate,test, columns_to_scale =
                                                       ['latitude','longitude','age','bath_bed_ratio', 'calc_sqft',
                                                       'taxrate','structure_dollar_per_sqft','tax_value'])
    #Make location cluster
    location = train_scaled[['latitude','longitude','age']]
    #Fit and predict location for k=4
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(location)
    kmeans.predict(location)
    #Add column back to train scaled for this cluster
    train_scaled['location_cluster'] = kmeans.predict(location)
    
    #Make Size cluster
    size = train_scaled[['bath_bed_ratio','calc_sqft']]
    #Fit and predict
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(size)
    kmeans.predict(size)
    #Add column back to train scaled for this cluster
    train_scaled['size_cluster'] = kmeans.predict(size)
    
    #Make value cluster
    value = train_scaled[['tax_value','structure_dollar_per_sqft']]
    #Fit and predict
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(value)
    kmeans.predict(value)
    #Add column back to train scaled for this cluster
    train_scaled['value_cluster'] = kmeans.predict(value)
    
    #Creating dummy variables from location cluster
    dummy_1 = pd.get_dummies(train_scaled.location_cluster, columns=['loc_0','loc_1','loc_2','loc_3'])
    # Concat these back onto df
    train_scaled = pd.concat([train_scaled, dummy_1], axis=1)
    #Rename clustered variables
    train_scaled.rename(columns = {0:'loc_0',1:'loc_1',2:'loc_2',3:'loc_3'}, inplace = True)
    
    #Creating dummy variables from size cluster
    dummy_2 = pd.get_dummies(train_scaled.size_cluster, columns=['size_0', 'size_1', 'size_2', 'size_3'])
    # Concat these back onto df
    train_scaled = pd.concat([train_scaled, dummy_2], axis=1)
    #Rename clustered variables
    train_scaled.rename(columns = {0:'size_0',1:'size_1',2:'size_2',3:'size_3'}, inplace = True)
    
    #Creating dummy variables from value cluster
    dummy_3 = pd.get_dummies(train_scaled.value_cluster, columns=['value_0','value_1','value_2','value_3'])
    # Concat these back onto df
    train_scaled = pd.concat([train_scaled, dummy_3], axis=1)
    #Rename clustered variables
    train_scaled.rename(columns = {0:'value_0',1:'value_1',2:'value_2',3:'value_3'}, inplace = True)
    
    return train_scaled, validate_scaled, test_scaled

