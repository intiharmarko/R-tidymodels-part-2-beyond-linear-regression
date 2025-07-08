#' 5 Final Project: Japan Real Estate Prices Prediction - Data pre-process - Functions


#' Load packages
library(tidyverse)


#' Parse date column
#'
#' This function merges year and quarter column into single date column, which is parsed
#' into date format column. After parsing, column holds the first of month for each
#' starting month of given quarter. Initial columns "year" and "quarter" are removed from df.
#' 
#' @param df A data frame holding the data.
#'
#' @return df A data frame with parsed date column.
#'
parse_date <- function(df = df.train){
  
  df <- df %>% 
    mutate(date_yq = paste0(year, "-", quarter),
           date = parse_date_time(date_yq, "y-q"),
           date = as_date(date)) %>% 
    select(-c("year", "quarter", "date_yq"))
  
  return(df)
}


#' Create columns list
#'
#' This function stores column names with column types into table.
#' Also target column is labelled. 
#' 
#' @param df  A data frame for given data.
#' @param tar A string holding the name of target column in df.
#'
#' @return df.cols A data frame of columns with column types added.
#'
create_cols_list <- function(df = df.train,
                             tar = "trade_price"){
  
  df.cols <- map(.x = df, 
                 .f = class) %>% 
    unlist() %>% 
    tibble(col = names(.),
           type = .) %>% 
    mutate(position = row_number(),
           role = if_else(col == tar, 
                          "target", 
                          "feature")) %>% 
    arrange(desc(role), position)
  
  return(df.cols)
}


#' Drop custom columns
#'
#' This function removes columns from given data frame. Columns (to drop) are 
#' provided as function input.
#' 
#' @param df A data frame holding the data.
#' @param cols A character vector holding names of the columns to be removed.
#'
#' @return df A data frame with columns removed.
#'
drop_cols_custom <- function(df = df.train,
                             cols){
  
  df <- df %>% 
    select(-all_of(cols))
  
  cat("Columns removed from data frame: ")
  cat(paste0(cols, collapse = ", "))
  
  return(df)
}


#' Drop columns with X % missing values
#'
#' This function removes columns from given data frame. Columns (to drop) are 
#' columns that have the number of missing rows above selected % threshold.
#' 
#' @param df A data frame holding the data.
#' @param cols A character vector holding names of the columns to be removed.
#'
#' @return df A data frame with columns removed.
#'
drop_cols_per_miss <- function(df = df.train,
                               per_t = 0.3,
                               cols.exc){
  
  # list columns with % of missing rows above threshold
  cols.drop <- map(.x = df,
                   .f = ~sum(is.na(.))) %>% 
    unlist() %>% 
    tibble(col = names(.),
           nr_missing = .,
           per_missing = nr_missing / nrow(df)) %>% 
    filter(per_missing >= per_t) %>% 
    pull(col)
  
  # exclude columns from exception list
  cols.drop <- cols.drop %>% setdiff(cols.exc)
  
  # remove columns from data frame
  df <- df %>% 
    select(-all_of(cols.drop))
  
  cat("Columns removed from data frame: ")
  cat(paste0(cols.drop, collapse = ", "))
  
  return(df)
}


#' Parse factor columns
#'
#' This function converts string categorical variables into factor type.
#' 
#' @param df A data frame holding the data (also string variables).
#'
#' @return df A data frame with parsed factor variables.
#'
parse_fct_cols <- function(df = df.train){
  
  df <- df %>% 
    mutate(across(where(is.character), as.factor))
  
  return(df)
}


#' Transform target (trade price) with log10 operation.
#'
#' This function transforms target variable trade price with log10 scale transformation.
#' There is an option to drop the initial column from the data set.
#' 
#' @param df A data frame holding the data.
#' @param drop_orig_col An logical - TRUE to drop original trade price column, FALSE to not drop.
#'
#' @return df A data frame with transformed trade price column.
#'
transform_tar_log10 <- function(df,
                                drop_orig_col = F){
  
  # apply transformation
  df <- df %>% 
    mutate(trade_price_log = log10(trade_price))
  
  # drop column if option selected
  if(drop_orig_col){
    df <- df %>% select(-trade_price)
  }
  
  return(df)
}


#' Extract year and quarter from reported transaction period.
#'
#' This function extracts two date components from date (reported period) column:
#' - year
#' - quarter of the year (potential indicator for season effect)
#' 
#' @param df A data frame holding the data.
#'
#' @return df A data frame with extracted date based features.
#'
extract_y_Q <- function(df){

  df <- df %>% 
    mutate(date_year = lubridate::year(date),
           date_Q = lubridate::quarter(date))
  
  return(df)
}


#' Calculate years since initial year.
#'
#' This function calculates feature that can potentially capture year based monotone 
#' drift that can affect trade price. Difference between observed year and initial year.
#' 
#' @param df A data frame holding the data.
#'
#' @return df A data frame with calculated years since initial year column.
#'
calculate_y_since <- function(df,
                              y_init = 2005){
  
  df <- df %>% 
    mutate(years_since_y_init = lubridate::year(date) - y_init)
  
  return(df)
}


#' Calculate quarters since initial quarter
#'
#' This function calculates feature that is difference between observed quarter
#' and initial quarter.
#' 
#' @param df A data frame holding the data.
#'
#' @return df A data frame with calculated quarters since initial quarter column.
#'
calculate_Q_since <- function(df,
                              Q_init = as.Date("2005-07-01")){
  
  df <- df %>% 
    mutate(Qs_since_Q_init = lubridate::time_length(lubridate::interval(Q_init, 
                                                                        date), 
                                                    "months") %/% 3)
  
  return(df)
}


#' Estimate min % per category to keep top X% of levels
#'
#' This function calculates percentage of rows each category - level holds for factor variable.
#' Percentages are sorted in descending order, and cumulative sum of percentages is added.
#' Categories are filtered based on cumulative percentage sum threshold, all levels
#' below the threshold are kept, and minimum percentage in kept categories is reported.
#' this value is input for step_other() later in recipe (modeling phase).
#' 
#' @param df A data frame holding the data.
#' @param fct_var A name of the factor feature.
#' @param per_thr A numeric (float) value representing cumulative sum percentage threshold.
#'
#' @return min_per A minimum percentage value for the last category in filtered counts - input for step_other().
#'
estimate_cat_min_per <- function(df = df.train,
                                 fct_var,
                                 per_thr){
  
  # categories counts & percentages already filtered
  counts <- df.train %>% 
    count(.data[[fct_var]]) %>% 
    filter(!is.na(.data[[fct_var]])) %>% 
    mutate(per = n / sum(n)) %>% 
    arrange(desc(n)) %>% 
    mutate(per_cs = cumsum(per)) %>% 
    filter(per_cs <= per_thr)
  
  # print counts
  print(counts)
  
  # store min percentage value
  min_per <- counts %>% pull(per) %>% min()
  
  cat("\nFactor feature: ", fct_var)
  cat("\nSelected cumulative sum percentage threshold: ", per_thr)
  cat("\nEstimated min percentage for the smallest category: ", min_per)
 
  return(min_per)
}
