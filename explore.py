import pandas as pd
import env
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from scipy import stats
from scipy.stats import levene, f_oneway
import warnings
warnings.filterwarnings('ignore')
import wrangle as w

def get_counties(df):
    '''
    This function will create dummy variables out of the original fips column. 
    And return a dataframe with all of the original columns except regionidcounty.
    We will keep fips column for data validation after making changes. 
    New columns added will be 'LA', 'Orange', and 'Ventura' which are boolean 
    The fips ids are renamed to be the name of the county each represents. 
    '''
    # create dummy vars of fips id
    county_df = pd.get_dummies(df.fips)
    # rename columns by actual county name
    county_df.columns = ['LA', 'Orange', 'Ventura']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    df_dummies = pd.concat([df, county_df], axis = 1)
    # drop regionidcounty and fips columns
    df_dummies = df_dummies.drop(columns = ['region_id_county'])
    return df_dummies


def create_features(df):
    '''This function creates new features based on the original features from the data.
    These features, including getting counties, are added back to the original dataframe'''
    #Getting Counties
    df = get_counties(df)
    #Creating age and binning from year built
    df['age'] = 2017 - df.year_built
    df['age_bin'] = pd.cut(df.age, 
                           bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
                           labels = [0, .066, .133, .20, .266, .333, .40, .466, .533, 
                                     .60, .666, .733, .8, .866, .933])

    # create taxrate variable
    df['taxrate'] = df.tax_amount/df.tax_value*100

    # create acres variable
    df['acres'] = df.lot_sqft/43560

    # bin acres
    df['acres_bin'] = pd.cut(df.acres, bins = [0, .10, .15, .25, .5, 1, 5, 10, 20, 50, 200], 
                       labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])

    # square feet bin
    df['sqft_bin'] = pd.cut(df.calc_sqft, 
                            bins = [0, 800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 7000, 12000],
                            labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                       )

    # dollar per square foot-structure
    df['structure_dollar_per_sqft'] = df.structure_tax_value/df.calc_sqft


    df['structure_dollar_sqft_bin'] = pd.cut(df.structure_dollar_per_sqft, 
                                             bins = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 1500],
                                             labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                            )


    # dollar per square foot-land
    df['land_dollar_per_sqft'] = df.land_value/df.lot_sqft

    df['lot_dollar_sqft_bin'] = pd.cut(df.land_dollar_per_sqft, bins = [0, 1, 5, 20, 50, 100, 250, 500, 1000, 1500, 2000],
                                       labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                      )


    # update datatypes of binned values to be float
    df = df.astype({'sqft_bin': 'float64', 'acres_bin': 'float64', 'age_bin': 'float64',
                    'structure_dollar_sqft_bin': 'float64', 'lot_dollar_sqft_bin': 'float64'})


    # ratio of bathrooms to bedrooms
    df['bath_bed_ratio'] = df.bath_count/df.bed_count

    # 12447 is the ID for city of LA. 
    # I confirmed through sampling and plotting, as well as looking up a few addresses.
    df['cola'] = df['region_id_city'].apply(lambda x: 1 if x == 12447.0 else 0)

    #Replace infinity values with nan
    df= df.replace([np.inf, -np.inf], np.nan)
    
    #Drop rows with null values created from calculations
    df = df.dropna()
    
    return df


####-----------------------Visualizations-----------------------###

# Visualization 1
def logerror_distribution(train):
    '''This function makes a histogram of the target variable, log error'''
    #Set Style
    sns.set_theme(style="darkgrid")
    # Set figure size
    plt.figure(figsize=(11,7))
    # plot it
    sns.histplot(data=train, x='log_error', color='#44cab9')
    #Limit the axes so we can see more clearly
    plt.xlim(-2,2)
    plt.ylim(0,50)
    plt.title('''Distribution of Log Error Appears Normal''')
    #Label the Axes
    plt.xlabel('Log Error')
    plt.ylabel('Count of Properties')
    plt.show()

    
# Visualization 2
def zestimate(train):
    '''This function makes a chart showing the difference in undervalued and overvalued Zestimates'''
    #Set figure size
    plt.figure(figsize=(11,7))
    #Create variable for value
    train['overvalue'] = train.log_error > 0
    #Make the plot
    ax = sns.histplot(train.overvalue, bins=3)
    # Add axes labels and ticks
    plt.xticks([0,1])
    plt.xlabel('Homes with greater or less than zero log error')
    plt.ylabel('Count of Homes')
    plt.title('Zestimate is more likely to overvalue a home')
    plt.show()
    
#Visualization 3
def loc_cluster_viz(train):
    '''This function visualizes Location Clusters with their features'''
    #Set Theme
    sns.set_theme()
    #Set Plot Size
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 7)
    #Make the Plot
    ax = sns.scatterplot(data=train, x="latitude", y="longitude", hue='location_cluster', palette='Blues')
    #Specify Axis labels
    ax.set(xlabel='Latitude',
        ylabel='Longitude',
        title='Location Cluster 1 is the Most Significant Driver of Log Error')
    #Hide scientific notation
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    plt.show()
    
#Visualization 4
def size_cluster_viz(train):
    '''This function visualizes size clusters with their features'''
    #Set Theme
    sns.set_theme()
    #Set Plot Size
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 7)
    #Make the Plot
    ax = sns.scatterplot(data=train, x="bath_bed_ratio", y="calc_sqft", hue='size_cluster', palette='PRGn')
    #Specify Axis labels
    ax.set(xlabel='Bathroom Bedroom Ratio',
        ylabel='Calculated Finished Square Feet',
        title='Size Cluster 3 Has a Significant Difference in Log Error')
    plt.show()
    
#Visualization 5

def value_cluster_viz(train):
    '''This function creates a visualization of Value features, hued by clusters'''
    #Set Theme
    sns.set_theme()
    #Set Plot Size
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 7)
    #Make the Plot
    ax = sns.scatterplot(data=train, x="tax_value", y="structure_dollar_per_sqft", hue='value_cluster', palette='Paired')
    #Specify Axis labels
    ax.set(xlabel='Tax Value ($) in ten millions',
        ylabel='Cost Per Square Feet ($)',
        title='Value Cluster 1 Has a Significant Difference in Log Error')
    #Hide scientific notation
    ax.xaxis.get_offset_text().set_visible(False)
    plt.show()
    
##------------------Statistical Testing------------------##

def location_ttest(train):
    '''This function creates a series from location cluster 1 and conducts a 1 sample
    T-test comparing the cluster log error to train mean log error'''
    overall_mean = train.log_error.mean()
    alpha = 0.05

    loc_cluster_one = train[train.location_cluster == 1].log_error
    population_mean = train.log_error.mean()

    t, p = stats.ttest_1samp(loc_cluster_one, overall_mean)

    print(f'Test Statistic: {t.round(2)}, P-Value: {p.round(2)}')
    
def size_ttest(train):
    '''This function creates a series from size cluster 3 and conducts a 1 sample
    T-test comparing the cluster log error to train mean log error'''
    overall_mean = train.log_error.mean()
    alpha = 0.05

    size_cluster_three = train[train.size_cluster == 3].log_error
    population_mean = train.log_error.mean()

    t, p = stats.ttest_1samp(size_cluster_three, overall_mean)

    print(f'Test Statistic: {t.round(2)}, P-Value: {p.round(2)}')
    
def value_ttest(train):
    '''This function creates a series from value cluster 1 and conducts a 1 sample
    T-test comparing the cluster log error to train mean log error'''
    overall_mean = train.log_error.mean()
    alpha = 0.05

    value_cluster_one = train[train.value_cluster == 1].log_error
    population_mean = train.log_error.mean()

    t, p = stats.ttest_1samp(value_cluster_one, overall_mean)

    print(f'Test Statistic: {t.round(2)}, P-Value: {p.round(2)}')