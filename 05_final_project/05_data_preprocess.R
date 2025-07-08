#' 5 Final Project: Japan Real Estate Prices Prediction - Data pre-process

rm(list = ls())
graphics.off()


# Load packages
library(tidyverse)
library(janitor)

# Load functions
source("./05_06_data_preprocess_functions.R")



# Data import

## real estate prices - train data
load("./data/train.RData")

## prefecture codes
df.pref_codes <- read_csv("./data/prefecture_code.csv", 
                          col_names = T)



# Data preprocess

## clean column names
df.train <- df.train %>% 
  clean_names()

df.pref_codes <- df.pref_codes %>% 
  clean_names()

## sample train data 
## - faster EDA / model training etc. 
## - (optional usage!)
set.seed(123)

df.s.train <- df.train %>% 
  sample_frac(size = 0.1)


## check data

### view df
View(df.train)

### dimensions
df.train %>% ncol()
df.train %>% nrow()

### show column names
df.train %>% colnames()


## parse date column
df.train <- parse_date(df.train)
df.s.train <- parse_date(df.s.train)


## create column list
## - column names
## - column types
df.cols <- create_cols_list() 


## remove columns
# - we are removing:
#   - column representing id rows of data frame
#   - column with remarks and time to nearest station
#   - columns with too many missing rows (values) ~ above certain threshold
#   - columns that can be used as alternative for price target column

## remove custom columns
cols.drop <- c("no", "remarks", 
               "time_to_nearest_station",
               "period",
               "unit_price", "price_per_tsubo")

df.train   <- drop_cols_custom(df = df.train,   cols = cols.drop)
df.s.train <- drop_cols_custom(df = df.s.train, cols = cols.drop)

## remove columns with high % of missing rows
## - some columns are added on exception list (not removed!)
## - selected % threshold is at least 40% missing rows
cols.exception <- c("building_year")

df.train   <- drop_cols_per_miss(df = df.train,   per_t = 0.4, cols.exc = cols.exception)
df.s.train <- drop_cols_per_miss(df = df.s.train, per_t = 0.4, cols.exc = cols.exception)


## parse column types
## - we parse string columns (categorical variables) -> converted to factors
df.train   <- parse_fct_cols(df.train)
df.s.train <- parse_fct_cols(df.s.train)



# Feature engineering
# - some feature engineering will be done on the fly (in this script)
# - other feature engineering will be done using recipe steps in modeling phase
# - here we will only list recipe steps based transformations for modeling phase

## extract year and quarter from transaction period (date column)
## - we extract year from date and quarter of the year (for potential seasonality capture)
df.train   <- extract_y_Q(df.train)
df.s.train <- extract_y_Q(df.s.train)

## calculate years since initial reported year (2005)
## - to capture potential year monotone drift 
## - that affects trade price
## - calculation: difference between observed year and initial year
df.train   <- calculate_y_since(df.train)
df.s.train <- calculate_y_since(df.s.train)

## calculate quarters since initial reported quarter (2005 Q3 ~ 2005-07-01)
## - to keep track how many quarters have gone by 
## - calculation: difference between observed quarter and initial quarter
df.train  <- calculate_Q_since(df.train)
df.s.train <- calculate_Q_since(df.s.train)


## dry run - recipe based feature engineering
## - actual transformations will be applied in modeling phase

### calculate squareness - how square is the shape of each real estate
### - squareness = frontage / sqrt(area)

### lump factor features (high categorical features)
### - categorical features with a large number of levels
### - will be lumped (keeping only top % of levels and lump remaining into "other" class)
### - usually we keep top 1% or 2% of categories
### - list of features for lumping
###   - "municipality"
###   - "district_name"
###   - "nearest_station"

### estimate min % per category to keep top X% of levels
mun_min_p <- estimate_cat_min_per(fct_var = "municipality", per_thr = 0.03)
dis_min_p <- estimate_cat_min_per(fct_var = "district_name", per_thr = 0.015)
nes_min_p <- estimate_cat_min_per(fct_var = "nearest_station", per_thr = 0.025)

### store min % in a list
list_min_p <- list(municipality = mun_min_p,
                   district_name = dis_min_p,
                   nearest_station = nes_min_p)

#*### create indication features that indicate which rows are missing for given feature

### impute missing values:
### - numeric predictors: median value imputation
### - factor predictors: missing class marker "missing"


## refresh column list
df.cols <- create_cols_list(tar = "trade_price_log") 



# Data export
# - export pre-processed training data
save(df.train, df.s.train, df.cols, list_min_p,
     file = "./data/train_preproc.RData")
