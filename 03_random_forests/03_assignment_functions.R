# Assignment - Random Forest for Ames data sale price prediction - Functions


#' Extract near-constant factor features 
#'
#' This function calculates percentage of rows for the most frequent level
#' in each categorical feature. And then it extracts all categorical
#' features where most dominant level rows percentage is above selected threshold!
#'
#' @param df A data frame used for feature extraction (normally train data)
#' @param per_thresh A numerical value - percentage threshold for feature extraction.
#'
#' @return feat_ncf A character vector, with feature names.
#'
near_const_fact_feat <- function(df, per_thresh){
  
  require(dplyr)
  
  feat_ncf <- df %>% 
    # select factor features
    select(where(is.factor)) %>% 
    pivot_longer(cols = everything(), 
                 names_to = "feature", 
                 values_to = "level") %>%
    # count rows per each level
    group_by(feature, level) %>%
    summarise(count = n(), 
              .groups = "drop") %>%
    group_by(feature) %>%
    # calculate percentage of rows
    mutate(per = count / sum(count)) %>%
    slice_max(order_by = count, 
              n = 1, 
              with_ties = FALSE) %>%
    ungroup() %>% 
    arrange(desc(per)) %>% 
    # keep only features with dominant level above selected percentage
    filter(per >= per_thresh) %>% 
    # extract only features 
    pull(feature)
  
  return(feat_ncf)
}



#' Visualize feature importance - top n features
#'
#' This function draws plot of top n features based on their estimated feature importance. 
#'
#' @param mod_fit A random forest model object (model fit). 
#' @param n A numerical value - number of top n features we visualize.
#'
#' @return None 
#'
plot_top_n_feat <- function(mod_fit, n){

  require(workflows)
  require(vip)
  
  mod_fit %>%
    extract_fit_parsnip() %>%
    vip(num_features = n) 
}



#' Extract top n features - feature importance
#'
#' This function extracts top n features, using estimated feature importance. 
#'
#' @param mod_fit A random forest model object (model fit). 
#' @param n A numerical value - number of top n features we extract.
#'
#' @return feat_top_n A character vector - with feature names.  
#'
extract_top_n_feat <- function(mod_fit, n){
  
  require(workflows)
  require(vip)
  
  feat_top_n <- rf_fit_fi %>%
    extract_fit_parsnip() %>%
    vi() %>%
    slice_max(Importance, 
              n = 20) %>% 
    pull(Variable)
  
  return(feat_top_n)
}
