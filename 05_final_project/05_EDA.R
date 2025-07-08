# 5 Final Project: Japan Real Estate Prices Prediction - EDA


# Load packages
library(tidyverse)

# Load functions
source("./05_05_EDA_functions.R")



# Initial EDA

## figure: column types count
col_type_count()

## figure: check missing values
rows_miss_count()

## figure: count data points (nr rows) per reported date (quarter)
rows_date_count()

## figure: count number of categories / levels - categorical (factors) features 
nr_categories_count()
nr_categories_count(n_max = 10000)
nr_categories_count(n_max = 2000)
nr_categories_count(n_max = 500)
nr_categories_count(n_max = 100)



# Target variable ("trade_price") distribution

## distribution target variable - time dimension flatten
## - draw distribution of price - density plot & box plot
## - time component not considered
target_var_distr(df.s.train, "dens")
target_var_distr(df.s.train, "box")

## distribution target variable - over time dimension 
## - draw distribution of price over time - scatter plot & box plot
## - time component is now considered
target_var_time_distr(p.type = "sct")
target_var_time_distr(p.type = "box")



# Numeric features distribution

## show numeric columns
df.cols <- create_cols_list()
df.cols %>% filter(role != "target" & type %in% c("numeric", "integer")) 

## "municipality code" distribution
plot_histogram(x_var = "municipality_code")

## "min" / "max" ~ "time_to_nearest_station" distribution
plot_histogram(x_var = "min_time_to_nearest_station")
plot_histogram(x_var = "max_time_to_nearest_station")

## "area" distribution
plot_density(x_var = "area")

## "area_is_greater_flag" distribution
plot_histogram(x_var = "area_is_greater_flag")
df.train %>% count(area_is_greater_flag)

## "frontage" distribution
plot_histogram(x_var = "frontage")

## "total_floor_area_is_greater_flag" distribution
plot_histogram(x_var = "total_floor_area_is_greater_flag")
df.train %>% count(total_floor_area_is_greater_flag)

## "building_year" distribution
plot_histogram(x_var = "building_year")

## "total_floor_area_is_greater_flag" distribution
plot_histogram(x_var = "prewar_building")
df.train %>% count(prewar_building)

## "breadth" distribution
plot_histogram(x_var = "breadth")

## "coverage_ratio" distribution
plot_histogram(x_var = "coverage_ratio")

## "floor_area_ratio" distribution
plot_histogram(x_var = "floor_area_ratio")



# Factor (categorical) features distribution

## show factor columns
df.cols %>% filter(role != "target" & type == "factor") 

## "type" distribution
plot_fct_dist(x_var = "type")

## "region" distribution
plot_fct_dist(x_var = "region")

## "prefecture" distribution
plot_fct_dist(x_var = "prefecture")

## "municipality" distribution
plot_fct_dist(x_var = "municipality", x_font_size = 1)

## "district_name" distribution
plot_fct_dist(df.s.train, x_var = "district_name", x_font_size = 1)

## "land_shape" distribution
plot_fct_dist(x_var = "land_shape")

## "direction" distribution
plot_fct_dist(x_var = "direction")

## "classification" distribution
plot_fct_dist(x_var = "classification")

## "city_planning" distribution
plot_fct_dist(x_var = "city_planning")

## "nearest_station" distribution
plot_fct_dist(x_var = "nearest_station", x_font_size = 1)



# Numeric features VS target variable - scatter plot

## "trade_price" ~ "municipality code" distribution
plot_tar_num_feat_dist(x_var = "municipality_code")

## "trade_price" ~ "min" / "max" ~ "time_to_nearest_station" distribution
plot_tar_num_feat_dist(x_var = "min_time_to_nearest_station")
plot_tar_num_feat_dist(x_var = "max_time_to_nearest_station")

## "trade_price" ~ "area" distribution
plot_tar_num_feat_dist(x_var = "area")

## "trade_price" ~ "area_is_greater_flag" distribution
plot_tar_num_feat_dist(x_var = "area_is_greater_flag")

## "trade_price" ~ "frontage" distribution
plot_tar_num_feat_dist(x_var = "frontage")

## "trade_price" ~ "total_floor_area_is_greater_flag" distribution
plot_tar_num_feat_dist(x_var = "total_floor_area_is_greater_flag")

## "trade_price" ~ "building_year" distribution
plot_tar_num_feat_dist(x_var = "building_year")

## "trade_price" ~ "total_floor_area_is_greater_flag" distribution
plot_tar_num_feat_dist(x_var = "prewar_building")

## "trade_price" ~ breadth" distribution
plot_tar_num_feat_dist(x_var = "breadth")

## "trade_price" ~ "coverage_ratio" distribution
plot_tar_num_feat_dist(x_var = "coverage_ratio")

## "trade_price" ~ "floor_area_ratio" distribution
plot_tar_num_feat_dist(x_var = "floor_area_ratio")



# Factor features VS target variable - box plots

## "trade_price" ~  "type" distribution
plot_tar_fct_feat_dist(data = df.s.train, x_var = "type")

## "trade_price" ~  "region" distribution
plot_tar_fct_feat_dist(data = df.s.train, x_var = "region")

## "trade_price" ~  "prefecture" distribution
plot_tar_fct_feat_dist(data = df.s.train, x_var = "prefecture")

## "trade_price" ~  "land_shape" distribution
plot_tar_fct_feat_dist(data = df.s.train, x_var = "land_shape")

## "trade_price" ~  "direction" distribution
plot_tar_fct_feat_dist(data = df.s.train, x_var = "direction")

## "trade_price" ~  "classification" distribution
plot_tar_fct_feat_dist(data = df.s.train, x_var = "classification")

## "trade_price" ~  "city_planning" distribution
plot_tar_fct_feat_dist(data = df.s.train, x_var = "city_planning")



# Correlation between numeric variables
# - calculate correlation between each pair of numeric variables (target included!)
# - and plot correlation heatmap
# - first refresh column list!

## column list refresh
df.cols <- create_cols_list()

## plot correlation heatmap
plot_corr_heatmap()
