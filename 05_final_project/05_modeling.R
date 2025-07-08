# 5 Final Project: Japan Real Estate Prices Prediction - Modeling

rm(list = ls())
graphics.off()

# Load packages
library(tidyverse)
library(tidymodels)
library(future)
library(doFuture)

library(rpart)
library(ranger)
library(xgboost)
library(bonsai)
library(lightgbm)


# Load functions
source("./05_09_modeling_functions.R")


# Data

## import data
load("./data/train_preproc.RData")

## split data (time based split)
## - training: "2005-07-01" ~ "2015-07-01" (41 quarters)
## - validate; "2015-10-01" ~ "2017-07-01" (8 quarters)
##
## - we will use sampled version of train data (also additionally sampled)
## - otherwise tuning step will take too long
set.seed(1123)
df.ss.train <- df.s.train %>% slice_sample(prop = 0.2, replace = F)

df.split.list <- split_data_date(data = df.ss.train, pivot_date = "2015-10-01")
df.training   <- df.split.list$training
df.validate   <- df.split.list$validate


## set up time based cross-validation for hyperpar. tuning
## - we will use time based folds
## - 20 quarters (5 years) for analysis (training)
## - 4 quarters (1 year) for assessment (estimate performance metric - RMSE)
## - 4 quarters (1 year) for sliding step in every iteration 
## - (we move 1 year in the future in the new fold relative to previous fold)
date_folds <- sliding_period(data = df.training,
                             index = "date",
                             period = "quarter",
                             lookback = 20,   
                             assess_stop = 4,
                             step = 4)

### inspect folds
fold <- 1 # 2 5
d_an <- analysis(date_folds$splits[[fold]])
d_as <- assessment(date_folds$splits[[fold]])
d_an %>% count(date) %>% as.data.frame()
d_as %>% count(date) %>% as.data.frame()



# Model training

## Model specifications
## - specify model specification
## - create recipe
## - create workflow
##
## - algorithms / models used and list of hyper-parameters we tune per model:
##   - XGBoost (engine - "xgboost")
##     - trees ~ number of trees (learners)
##     - tree_depth ~ max depth of each tree
##     - learn_rate ~ learning rate in boosting step
##   - LigthGBM (engine - "ligthgbm")
##     - trees ~ number of trees (learners)
##     - tree_depth ~ max depth of each tree
##     - learn_rate ~ learning rate in boosting step 

### XGBoost model
xgb_mod <- boost_tree(mode = "regression",
                      trees = tune(),
                      tree_depth = tune(),
                      learn_rate = tune()) %>%
  set_engine("xgboost")

### LigthGBM model
lgbm_mod <- boost_tree(mode = "regression",
                       trees = tune(),
                       tree_depth = tune(),
                       learn_rate = tune()) %>%
  set_engine("lightgbm")

### List of models specs.
mod_list <- list(xgb = xgb_mod,
                 lgbm = lgbm_mod)


## Recipe
## - we will apply identical recipe steps for all algorithms
## - to assure the same pre-processed data will be used for all algorithms
## - recipe steps applied:
##   - create new feature: "squareness" (squareness = frontage / sqrt(area m2))
##   - create indicator features for all missing rows (all features with any missing rows)
##   - impute missing numeric features with median values
##   - collapse infrequent categorical levels for factor features with high cardinality categories
##   - impute missing factor features with unknown category
##   - one-hot encode categorical features
## - also we do not include date column in modeling (will cause errors)

## create recipe
rec <- recipe(formula = trade_price ~ .,
              data = df.training %>% select(-date)) %>%
  step_mutate(squareness = frontage / sqrt(area)) %>% 
  step_integer(all_logical(), zero_based = TRUE) %>% 
  step_other(municipality, threshold = list_min_p$municipality) %>% 
  step_other(district_name, threshold = list_min_p$district_name) %>% 
  step_other(nearest_station, threshold = list_min_p$nearest_station) %>% 
  step_unknown(all_nominal_predictors(), new_level = "missing") %>%
  step_impute_median(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())

### inspect recipe steps
rec

prep_rec <- prep(rec, retain = T, verbose = T)
tidy(prep_rec)
r_step1 <- tidy(prep_rec, number = 1)
r_step2 <- tidy(prep_rec, number = 2)
r_step3 <- tidy(prep_rec, number = 3)
r_step4 <- tidy(prep_rec, number = 4)
r_step5 <- tidy(prep_rec, number = 5)
r_step6 <- tidy(prep_rec, number = 6)
r_step7 <- tidy(prep_rec, number = 7)
r_step8 <- tidy(prep_rec, number = 8)

### list of recipes
rec_list <- list(xgb = rec,
                 lgbm = rec)


## Workflows
## - match model specifications with recipes
## - create workflow set
wflow_set <- workflow_set(
  preproc = rec_list,
  models = mod_list,
  cross = F) %>% 
  mutate(wflow_id = names(mod_list))


## Finalize model's parameters (features) list
## - we need to provide features list
## - for each combination of model and used recipe
## - for tuning step
par_fin_list <- finalize_model_params()


## hyperpar. tuning
## - execute tuning using date based cross-validation data
## - tuning is executed using Bayesian optimization 
## - Bayes optimization tuning specs:
##   - 4 initial points 
##   - max 10 iterations
##   - stop if no improvements after 3 iterations
## - we will tune using parallel computing on multiple CPU-s
## - we must register backend for foreach  
## - we must specify cores for parallel computing

registerDoFuture()                                        # parallel backend for foreach
plan(multisession, workers = parallel::detectCores() - 1) # use multi-cores
future::nbrOfWorkers()                                    # check how many workers are active

set.seed(235)

tune_rez <- tune_bayes_custom(cv_folds_ = date_folds,
                              iter = 10, 
                              initial = 4, 
                              no_improve = 3)


save_WS_snapshot("./data/tuning_rez.RData")
#load_WS_snapshot("./data/tuning_rez.RData")


## select best model (per each algorithm!) & finalize it
## - model with lowest RMSE
## - and selected values for hyperpar.
## - finalize workflow with best model
## - train model (model fit) on whole train data
## - for all selected algorithms
mod_fit_list <- fit_best_models(df.train_ = df.training)



# Model Validation - Select best performing algorithm 
# - evaluate each models performance
# - on validation data (RMSE metrics)
# - first predict output on validation data
# - then extract RMSE
# - visualize RMSE & select best performing algorithm

## predict and prepare the prediction data frames
df.validate_pred <- map(mod_fit_list, ~ predict(.x, df.validate) %>% 
                          bind_cols(df.validate) %>% 
                          select(trade_price, .pred)) 

## calculate RMSE for each model
validate_rmse_list <- map(df.validate_pred, 
                          ~ metrics(.x, 
                                    truth = trade_price, 
                                    estimate = .pred) %>% 
                            filter(.metric == "rmse"))

## visualize RMSE (validation set)
## - compare validation RMSE for all models (different algorithm types - best model candidate)
bind_rows(validate_rmse_list, .id = "algorithm") %>% 
  ggplot(aes(x = algorithm,
             y = .estimate,
             fill = algorithm,
             label = round(.estimate, 3))) +
  geom_col(color = "black",
           show.legend = F) +
  geom_text(size = 10) +
  labs(title = "Different algorithms' prediction performance: RMSE validation set",
       x = "Algorithm",
       y = "RMSE") +
  theme_minimal(base_size = 16)

## select best performing algorithm
alg_best <- bind_rows(validate_rmse_list, .id = "algorithm") %>% 
  arrange(.estimate) %>% 
  dplyr::slice(1) %>% 
  pull(algorithm)

print(paste0("Best performing algorithm: ", alg_best))


## Visualize training results for final selected algorithm

## Visualize tuning results
## - draw tuning hyperparameter results
## - each hyperparameter placed in its own facet
## - use facet grid to create facets for each hyperparameter 
## - each hyperparameter value VS RMSE
## - use tuning results (during training - CV)

alg_best_name <- get_alg_full_name() # get full algorithm name for figure

tune_rez[[alg_best]] %>% 
  collect_metrics() %>% 
  filter(.metric == "rmse") %>% 
  select(trees,
         tree_depth,
         learn_rate,
         mean, 
         std_err) %>% 
  pivot_longer(cols = trees:learn_rate,
               names_to = "par",
               values_to = "par_value") %>% 
  ggplot(aes(x = par_value, 
             y = mean)) +
  geom_point(size = 4, 
             color = "brown3") +
  geom_line(color = "gray20", 
            linewidth = 1) +
  geom_errorbar(aes(ymin = mean - std_err, 
                    ymax = mean + std_err), 
                width = 0.5, 
                alpha = 0.5) +
  facet_grid(cols = vars(par),
             scales = "free",
             labeller = "label_both") +
  labs(title = paste0("Alg. ",  alg_best_name, " tuning: RMSE vs. hyperparameter values"),
       x = "Hyperparameter value",
       y = "RMSE") +
  theme_minimal(base_size = 16)

### show values for hyperparam. selected final model
mod_fit_list[[alg_best]]$fit$actions$model$spec


## Train final model using whole train set 
## - here we use training + validate data set
## - whole train data is used to estimate model parameters
## - this model will be used for final price predictions (using test set  - unseen data)
mod_fin_fit <- fit_fin_model(df.train_ = df.ss.train)



# Model Testing 
# - best performing algorithm is tested on unseen test data
# - import test data 
# - then apply feature engineering steps (done in data preprocess script!)
# - use model to predict trade_price_log on test data
# - then extract RMSE
# - convert log price predictions to price 
# - visualize price vs predicted price

## load data pre-processing functions
source("./05_06_data_preprocess_functions.R")


## load data - test data
load("./data/test.RData")


## apply data pre-processing steps & feature engineering

### clean column names
df.test <- df.test %>% 
  janitor::clean_names()

### parse date column
df.test <- parse_date(df.test)

### parse column types
df.test <- parse_fct_cols(df.test)

### extract year and quarter from transaction period (date column)
df.test <- extract_y_Q(df.test)

### calculate years since initial reported year (2005)
df.test <- calculate_y_since(df.test)

### calculate quarters since initial reported quarter (2005 Q3 ~ 2005-07-01)
df.test <- calculate_Q_since(df.test)


## predict price and add predictions to test data frame
df.test <- predict(mod_fin_fit, df.test) %>% 
  bind_cols(df.test) %>% 
  rename(pred_trade_price = .pred) %>% 
  select(trade_price, pred_trade_price, everything())

## calculate test RMSE
test_rmse <- df.test %>% 
  metrics(truth = trade_price, 
          estimate = pred_trade_price) %>% 
  filter(.metric == "rmse")

## draw actual prices VS predicted
set.seed(1123)

df.test %>% 
  slice_sample(prop = 0.1) %>% 
  ggplot(aes(x = trade_price,
             y = pred_trade_price)) +
  geom_point() +
  ggtitle("Predicted prices VS actual prices") +
  xlab("Predicted trade price (in yen)") +
  ylab("Actual trade price (in yen)") +
  theme_minimal(base_size = 16)
