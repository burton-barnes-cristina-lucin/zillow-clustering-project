import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import os
import numpy as np
import env

from env import user, password, host

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