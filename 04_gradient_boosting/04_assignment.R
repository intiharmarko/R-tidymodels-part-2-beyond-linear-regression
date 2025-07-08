# Assignment - Selected ML algorithms for Ames data sale price prediction

rm(list = ls())
graphics.off()

# Install packages
#install.packages("bonsai")
#install.packages("lightgbm")

# Load packages
library(tidyverse)
library(tidymodels)
library(future)
library(doFuture)
library(janitor)

library(rpart)
library(ranger)
library(xgboost)
library(bonsai)
library(lightgbm)


# Load functions
source("./04_assignment_functions.R")


# Data

## Ames housing prices
df <- modeldata::ames %>% 
  janitor::clean_names()

## split data
##
## - train    70%
##   - CV (k = 10)
## - validate 20%
## - test     10%
set.seed(1123)

df.split.list <- split_data(df, p_t = 7/10, p_v = 2/3)
df.train    <- df.split.list$train
df.validate <- df.split.list$validate
df.test     <- df.split.list$test

## set up CV for hyperpar. tuning
## - we will use k=10 fold CV
cv_folds <- vfold_cv(df.train, v = 10)


# Model training

## Model specifications
## - specify model specification
## - create recipe
## - create workflow
##
## - algorithms / models used and list of hyper-parameters we tune per model:
##   - KNN (engine - "kknn")
##     - neighbors ~ K (number of neighbors)
##   - Decision tree (engine - "rpart")
##     - min_n ~ min number of samples per leaf 
##     - tree_depth ~ max number of tree's splitting levels
##     - cost_complexity ~ alpha parameter that controls tree complexity via pruning
##   - Random forest (engine - "ranger")
##     - mtry ~ number of predictors (features) sampled at each split
##     - min_n ~ min number of observations per node
##     - trees ~ number of trees in the forest
##   - XGBoost (engine - "xgboost")
##     - trees ~ number of trees (learners)
##     - tree_depth ~ max depth of each tree
##     - learn_rate ~ learning rate in boosting step
##   - LigthGBM (engine - "ligthgbm")
##     - trees ~ number of trees (learners)
##     - tree_depth ~ max depth of each tree
##     - learn_rate ~ learning rate in boosting step 

### KNN model
knn_mod <- nearest_neighbor(mode = "regression",
                            neighbors = tune()) %>%
  set_engine("kknn")

### Decision tree model
tree_mod <- decision_tree(mode = "regression",
                          min_n = tune(),
                          tree_depth = tune(),
                          cost_complexity = tune()) %>%
  set_engine("rpart")

### Random Forest model
rf_mod <- rand_forest(mode = "regression",
                      mtry = tune(),
                      min_n = tune(),
                      trees = tune()) %>%
  set_engine("ranger")

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
mod_list <- list(knn = knn_mod,
                 tree = tree_mod,
                 rf = rf_mod,
                 xgb = xgb_mod,
                 lgbm = lgbm_mod)


## Recipes
## - some recipes specifics are per algorithm
##   - no pre-processing needed: decision trees & random forests
##   - encoding categorical features needed: KNN, XGBoost & LightGBM
##   - normalizing numeric features needed: KNN
##   - removing features with zero variance (problem will occur later near ZV numeric features): KNN

### basic recipe - no pre-processing
rec_basic <- recipe(formula = sale_price ~ .,
                    data = df.train)

### recipe with dummy features 
rec_dummy <- rec_basic %>% 
  step_dummy(all_nominal_predictors()) 

### recipe with dummy features + removed zero variance features + normalized numeric features 
rec_norm_nzv_dummy <- rec_dummy %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())
  
### list of recipes
rec_list <- list(knn = rec_norm_nzv_dummy,
                 tree = rec_basic,
                 rf = rec_basic,
                 xgb = rec_dummy,
                 lgbm = rec_dummy)


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
## - execute tuning using CV data
## - tuning is executed using Bayesian optimization 
## - Bayes optimization tuning specs:
##   - 5 initial points 
##   - max 20 iterations
##   - stop if no improvements after 5 iterations
## - we will tune using parallel computing on multiple CPU-s
## - we must register backend for foreach  
## - we must specify cores for parallel computing

registerDoFuture()                                        # parallel backend for foreach
plan(multisession, workers = parallel::detectCores() - 1) # use multi-cores
future::nbrOfWorkers()                                    # check how many workers are active

set.seed(235)

tune_rez <- tune_bayes_custom(iter = 20, 
                              initial = 5, 
                              no_improve = 5)


## select best model (per each algorithm!) & finalize it
## - model with lowest RMSE
## - and selected values for hyperpar.
## - finalize workflow with best model
## - train model (model fit) on whole train data
## - for all selected algorithms
mod_fit_list <- fit_best_models()



# Model Validation - Select best performing algorithm 
# - evaluate each models performance
# - on validation data (RMSE metrics)
# - first predict output on validation data
# - then extract RMSE
# - visualize RMSE & select best performing algorithm

## predict and prepare the prediction data frames
df.validate_pred <- map(mod_fit_list, ~ predict(.x, df.validate) %>% 
                      bind_cols(df.validate) %>% 
                      select(sale_price, .pred))

## calculate RMSE for each model
validate_rmse_list <- map(df.validate_pred, 
                      ~ metrics(.x, 
                                truth = sale_price, 
                                estimate = .pred) %>% 
                        filter(.metric == "rmse"))

## visualize RMSE (validation set)
## - compare validation RMSE for all models (different algorithm types - best model candidate)
bind_rows(validate_rmse_list, .id = "algorithm") %>% 
  ggplot(aes(x = algorithm,
             y = .estimate,
             fill = algorithm,
             label = round(.estimate,1))) +
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
mod_fit_list[[alg_best]]$fit$actions$model$spec



# Model Testing
# - evaluate best model performance 
# - on test data (RMSE metrics)
# - first predict output on test data
# - then extract RMSE

# predict on test set
df.test_best_pred <- predict(mod_fit_list[[alg_best]], df.test) %>% 
  bind_cols(df.test) %>% 
  select(sale_price, .pred)

# calculate test RMSE
test_best_rmse <- df.test_best_pred %>% 
  metrics(truth = sale_price, 
          estimate = .pred) %>% 
  filter(.metric == "rmse")

test_best_rmse
