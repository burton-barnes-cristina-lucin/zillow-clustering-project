# Zillow Clustering Project
## Project Description

Zillow is a billion dollar real estate business that sells thousands of houses in the state of California alone. We have decided to look into log error, the overvaluation or undervalution of a houses' worth. We will look into what features if any influence the amount of log error and create a model to estimate the log error for properties.

# Project Goal

* Discover driver of log error from the Zillow data
* Use drivers to develop machine learning model to predict log error
* This information will be used to further our understanding of which elements contribute to or detract from a log_error.

# Initial Thoughts

* Log Error is driven by certain home features, such as size, location, or value metrics.

# The Plan

* Aquire data

* Explore data in search of what causes log_error
    * Answer the following initial questions
        * What is the distribution of logerror?
        * Is a property more likely to over evaluated or under evaluated?
        * If we cluster location data with home age, is there a relationship with log error?
        * If we cluster size features(Bath bed ratio and calculated finished square feet), is there a relationship with log error?
        * If we cluster value features(tax value, structure dollar square feet), is there a relationship with log error?
        
* Develop a Model to predict log error
    * Use drivers identified in explore to build predictive models of different types
    * Evaluate models on train and validate data
    * Select the best model to use on test data
    
* Draw Conclusions

# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|fips| Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more detail|
|latitude| Latitude of the middle of the parcel multiplied by 10e6|
|longitude| Longitude of the middle of the parcel multiplied by 10e6|
|LA| fips for the LA county|
|Orange| fips for the Orange county|
|Ventura| fips for the Ventura county|
|yearbuilt| The Year the principal residence was built|
|age| The year sold, 2017, minus the yearbuilt|
|age_bin| The age of the residence divided into several bins|
|taxamount| The total property tax assessed for that assessment year|
|taxrate| The taxamount divided by tax value multiplied by 100|
|taxvalue| The total tax assessed value of the parcel|
|lot_sqft| Area of the lot in square feet|
|acres| lot_sqft divided by 43560|
|acres_bin| The acres of residence divided into several bins|
|sqft_bin| The sqft of residence divided into several bins|
|structure_dollar_per_sqft| The tax value divided by sqft|
|structure_dollar_sqft_bin| A division of the structure dollar divided into several bins|
|land_dollar_per_sqft| land_value divided by lot_sqft|
|lot_dollar_sqft_bin| land_dollar_per_sqft divided into several bins|
|bath_count| number of bathrooms in residence|
|bed_count| number of bedrooms in residence|
|bath_bed_ratio| bath_count divided by bed_count|
|cola| Whether or not a residence is in the city of LA|

# Steps to Reproduce
1) Clone this repository
2) Ensure an .env file is in the repo, with definitions of host, username, and password for the Codeup database
2) Aquire the data from Codeup SQL database
3) Put the data in  the file containing the cloned repo
4) Run notebook

# Takeaways and Conclusions
* Feature engineering suburban vs. urban areas is likely to provide value to a model
* Feature engineering neighborhoods within the three counties is likely to improve modeling
* Additional creation of clusters could provide more insight into log error

# Next Steps
* Better data collection from real estate agents and brokers will help avoid missing values, which plague this dataset
* Features that include amenities that are not part of a home, such as crime rate, distance from hospitals, and school districts are likely to add value to a model