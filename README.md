# Zillow Clustering Project
## Project Description

Zillow is a billion dollar real estate business that sells thousands of houses in the state of California alone. We have decided to look into log_error, the overvaluation or undervalution of a houses worth. We will look into what features if any influence the amount of log_error and create a model to estimate the houses value.

# Project Goal

* Discover driver of log_error from the zillow database.
* Use drivers to develop machine learning model to predict log_error
* This information will be used to further our understanding of which elements contribute to or detract from a log_error.

# Initial Thoughts

The initial thoughts are that people overvalue their house and that

# The Plan

* Aquire data

* Explore data in search of what causes log_error
    * Answer the following initial questions
        * 
        * 
        * 
        * 
        * 
        
* Develop a Model to predict the log_error
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
1) Clone this repo.
2) Aquire the data from SQL
3) Put the data in  the file containing the cloned repo.
4) Run notebook.

# Takeaways and Conclusions
*
*
*

# Recommendations
*
*
*