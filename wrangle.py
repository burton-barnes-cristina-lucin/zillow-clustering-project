import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import os
import numpy as np
import env

from env import user, password, host

#----------------------Acquire the Data-----------------------#

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def acquire_zillow():
    ''' Retrieve data from Zillow database within codeup, selecting specific features
    If data is not present in directory, this function writes a copy as a csv file. 
    '''
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        query = '''SELECT *
        FROM properties_2017
        LEFT JOIN predictions_2017 USING (parcelid)
        LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
        LEFT JOIN buildingclasstype USING (buildingclasstypeid)
        LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
        LEFT JOIN airconditioningtype USING (airconditioningtypeid)
        LEFT JOIN storytype USING (storytypeid)
        LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
        LEFT JOIN propertylandusetype USING (propertylandusetypeid)
        LEFT JOIN unique_properties USING (parcelid)
        WHERE transactiondate LIKE '2017%%'
        AND latitude is not NULL
        AND longitude is not NULL
        AND (propertylandusetypeid = 261 OR propertylandusetypeid = 279);'''
        df = pd.read_sql(query, get_connection('zillow'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)

        # Return the dataframe to the calling code
        return df
    

##------------------------------Prepare the Data-----------------------------##

def handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75):
    '''This function drop rows or columns based on the percent of values that are missing,
    dropping columns before rows'''
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    return df

def prepare_zillow(df):
    '''This function imputes missing values when applicable, and drops columns with too much missing data. Finally,
    it utilizes another function to handle the missing values based on the proportion of rows and column values missing'''
    df.fullbathcnt = df.fullbathcnt.fillna(0)
    df.pooltypeid2 = df.pooltypeid2.fillna(0)
    df.pooltypeid10 = df.pooltypeid10.fillna(0)
    df.pooltypeid7 = df.pooltypeid7.fillna(0)
    df.fireplacecnt = df.fireplacecnt.fillna(0)
    df.decktypeid = df.decktypeid.fillna(0)
    df.poolcnt = df.poolcnt.fillna(0)
    df.hashottuborspa = df.hashottuborspa.fillna(0)
    df.typeconstructiondesc = df.typeconstructiondesc.fillna('None')
    df.fireplaceflag = df.fireplaceflag.fillna(0)
    df.threequarterbathnbr = df.threequarterbathnbr.fillna(0)
    df.taxdelinquencyyear = df.taxdelinquencyyear.fillna(99999)
    df.taxdelinquencyflag = df.taxdelinquencyflag.fillna('N')
    df.calculatedbathnbr = df.calculatedbathnbr.fillna(0)
    df.basementsqft = df.basementsqft.fillna(0)
    df.numberofstories.value_counts(dropna=False)
    
    #handle missing values
    df = handle_missing_values(df, prop_required_columns=.5, prop_required_rows=.75)
    
    #Drop columns with too many null values/extraneous information
    df = df.drop(columns=['buildingqualitytypeid','propertyzoningdesc','unitcnt','heatingorsystemdesc','id', 'id.1','heatingorsystemtypeid'])
    
    #Replace a whitespace sequence or empty with a NaN value and reassign this manipulation to df. 
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    #Drop remainder of rows with null values
    df = df.dropna()
    
    #Rename columns to something readable
    rename_dict = {
    'parcelid':'parcel_id', 'basementsqft':'basement_sqft',
    'bathroomcnt':'bath_count', 'bedroomcnt':'bed_count',
       'calculatedbathnbr':'calc_bath_and_bed','finishedfloor1squarefeet':'finished_floor1_sqft',
       'calculatedfinishedsquarefeet':'calc_sqft', 'finishedsquarefeet12':'finished_sqft12',
       'finishedsquarefeet13':'finished_sqft13', 'finishedsquarefeet15':'finished_sqft15',
        'finishedsquarefeet50':'finished_sqft50',
       'finishedsquarefeet6':'finished_sqft6', 'fireplacecnt':'fireplace_cnt',
        'fullbathcnt':'full_bath_cnt',
       'garagecarcnt':'garage_car_count', 'garagetotalsqft':'garage_sqft',
       'hashottuborspa':'has_hot_tub',
        'lotsizesquarefeet':'lot_sqft', 'poolcnt':'pool_count', 'poolsizesum':'sum_pool_size',
        'propertycountylandusecode':'property_county_use_code',
        'propertyzoningdesc':'property_zoning_desc',
       'rawcensustractandblock':'raw_census_tract_block', 'regionidcity':'region_id_city',
        'regionidcounty':'region_id_county',
       'regionidneighborhood':'region_id_neighbor', 'regionidzip':'region_id_zip',
        'roomcnt':'room_count', 'threequarterbathnbr':'three_quarter_bath',
       'unitcnt':'unit_count', 'yardbuildingsqft17':'yard_building_sqft17',
        'yardbuildingsqft26':'yard_building_sqft26', 'yearbuilt':'year_built',
       'numberofstories':'no_stories', 'fireplaceflag':'fireplace_flag',
        'structuretaxvaluedollarcnt':'structure_tax_value',
       'taxvaluedollarcnt':'tax_value', 'assessmentyear':'assessment_year',
        'landtaxvaluedollarcnt':'land_value',
       'taxamount':'tax_amount', 'taxdelinquencyflag':'tax_delinquency_flag',
       'taxdelinquencyyear':'tax_delinquency_year',
       'censustractandblock':'census_tract_and_block', 'logerror':'log_error',
       'transactiondate':'transaction_date',
       'airconditioningdesc':'air_conditioning_desc',
       'architecturalstyledesc':'architectural_style_desc',
       'buildingclassdesc':'building_class_desc',
       'heatingorsystemdesc':'heating_system_desc', 'propertylandusedesc':'property_land_use_desc',
        'storydesc':'story_desc',
       'typeconstructiondesc':'type_construction_desc'
                }
    df = df.rename(columns=rename_dict)
    
    return df  



##------------------------------Train Test split-----------------------------##

def train_validate_test_split(df, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 15% of the original dataset, validate is .1765*.85= 15% of the 
    original dataset, and train is 70% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.15, 
                                            random_state=seed)
    train, validate = train_test_split(train_validate, test_size=0.1765, 
                                       random_state=seed)
    return train, validate, test