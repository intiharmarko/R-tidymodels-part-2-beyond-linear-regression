# Assignment - Random Forest for Ames data sale price prediction

rm(list = ls())
graphics.off()

# Install packages
#install.packages("vip")

# Load packages
library(tidyverse)
library(tidymodels)
library(future)
library(doFuture)
library(ranger)
library(janitor)
library(vip)

# Load functions
source("./03_assignment_functions.R")


# Data

## Ames housing prices
df <- modeldata::ames %>% 
  janitor::clean_names()

## split data
set.seed(1123)

split <- initial_split(df, prop = 0.8)
df.train <- training(split)
df.test  <- testing(split)

## set up CV for hyperpar. tuning
## - we will use k=10 fold CV
cv_folds <- vfold_cv(df.train, v = 10)


# Model Training - Model 1 
#
# - all features used
# - tune hyper parameters using CV

## define model specification
## - "ranger" algorithm
## - we tune (hyperpar.):
##   - mtry ~ number of predictors (features) sampled at each split
##   - min_n ~ min number of observations per node
##   - trees ~ number of trees in the forest

rf1_mod <- rand_forest(mode = "regression",
                       mtry = tune(),
                       min_n = tune(),
                       trees = tune()) %>%
  set_engine("ranger")

## define recipe
## - we predict sale price using all remaining columns (features)
rf1_rec <- recipe(formula = sale_price ~ .,
                  data = df.train) 

## create workflow 
rf1_wflow <- workflow() %>%
  add_recipe(rf1_rec) %>%
  add_model(rf1_mod)

## hyperpar. tuning
## - execute tuning using CV data
## - tuning is executed using Bayesian optimization 
## - Bayes optimization tuning specs:
##   - 5 initial points 
##   - max 20 iterations
## - we will tune using parallel computing on multiple CPU-s

registerDoFuture()                                        # parallel backend for foreach
plan(multisession, workers = parallel::detectCores() - 1) # use multi-cores
future::nbrOfWorkers()                                    # check how many workers are active

set.seed(235)

### extract and finalize the parameter set
### - we need to finalize the parameter set, 
par1_final <- extract_parameter_set_dials(rf1_mod) %>%
  finalize(df.train %>% select(-sale_price))

rf1_tune_rez <- tune_bayes(
  rf1_wflow,
  resamples = cv_folds,
  metrics = metric_set(rmse),
  initial = 5,                              # initial grid points
  iter = 20,                                # number of optimization iterations
  param_info = par1_final,                  # hyperpar. to tune
  control = control_bayes(verbose = TRUE,
                          parallel_over = "everything")
)


## select best model 
## - model with lowest RMSE
## - and selected value for hyperpar.
rf1_mod_best <- select_best(rf1_tune_rez, metric = "rmse")

## finalize model
## - finalize workflow with best model
## - train model (model fit) on whole train data
rf1_wflow_fin <- finalize_workflow(rf1_wflow, rf1_mod_best)
rf1_mod_fit <- fit(rf1_wflow_fin, df.train)


# Model Training - Model 2 
#
# - reduced feature space using classical statistical techniques:
#    - near zero variance features removal
#    - highly correlated numeric features removal
#    - categorical features (one level too high percentage of rows) removal
# - tune hyper parameters using CV for model fit

## define model specification
## - model specification identical to model 1
rf2_mod <- rf1_mod


## define recipe
## - step before recipe is applied: 
##   - create a list of near-constant factor features (to be removed) ~ no recipe step available
## - recipe step: remove near-zero variance features
## - recipe step: remove highly correlated numeric features
## - predict sale price using all features that are not removed!

### percentage thresholds
ncff_per <- .95 # percentage threshold for removal near-constant factor features
corr_per <- .8  # coefficient correlation threshold for removal (absolute correlation value)

### list of near-constant factor features (to be removed)
ncff <- near_const_fact_feat(df.train, ncff_per)

### create recipe
rf2_rec <- recipe(formula = sale_price ~ .,
                  data = df.train %>% select(-one_of(ncff))) %>% 
  step_nzv(all_predictors()) %>% 
  step_corr(all_numeric_predictors(), threshold = corr_per)

### show removed features by each removal step  
ncff                      # show removed features by ncff
prep(rf2_rec) %>% tidy(1) # show removed features by nzv
prep(rf2_rec) %>% tidy(2) # show removed features by corr


## create workflow 
rf2_wflow <- workflow() %>%
  add_recipe(rf2_rec) %>%
  add_model(rf2_mod)

## hyperpar. tuning

### extract transformed training set
### - we need to extract training set without removed features
### - we use it in hyper parameters finalize step
df.train.processed <- rf2_rec %>% 
  prep() %>% 
  juice()

### extract and finalize the parameter set
### - we need to finalize the parameter set (mtry parameter)
par2_final <- extract_parameter_set_dials(rf2_mod) %>%
  finalize(df.train.processed)

rf2_tune_rez <- tune_bayes(
  rf2_wflow,
  resamples = cv_folds,
  metrics = metric_set(rmse),
  initial = 5,                              # initial grid points
  iter = 20,                                # number of optimization iterations
  param_info = par2_final,                  # hyperpar. to tune
  control = control_bayes(verbose = TRUE,
                          parallel_over = "everything")
)


## select best model 
## - model with lowest RMSE
## - and selected value for hyperpar.
rf2_mod_best <- select_best(rf2_tune_rez, metric = "rmse")

## finalize model
## - finalize workflow with best model
## - train model (model fit) on whole train data
rf2_wflow_fin <- finalize_workflow(rf2_wflow, rf2_mod_best)
rf2_mod_fit <- fit(rf2_wflow_fin, df.train)



# Model Training - Model 3 
#
# - estimate feature importance using random forest
#   - do not use CV for feature importance estimation
#   - visualize feature importance
# fit model with top n features (n = 20)
# - tune hyper parameters using CV for model fit

## feature importance estimation - model
## - fixed number of trees (500)
## - default values for hyper parameters
rf_mod_fi <- rand_forest(mode = "regression", 
                         trees = 500) %>%
  set_engine("ranger", importance = "permutation")

rf_rec_fi <- recipe(sale_price ~ ., 
                    data = df.train)

rf_wf_fi <- workflow() %>%
  add_recipe(rf_rec_fi) %>%
  add_model(rf_mod_fi)

rf_fit_fi <- fit(rf_wf_fi, data = df.train)

### visualize feature importance - top 20 features
plot_top_n_feat(rf_fit_fi, 20)

### extract top n = 20 features
feat_top_n <- extract_top_n_feat(rf_fit_fi, 20)


## define model specification
## - model specification identical to model 1
rf3_mod <- rf1_mod


## define recipe actual mode we build
## - use only top n = 20 features based on feature importance
## - predict sale price using all features that are not removed!
rf3_rec <- recipe(formula = sale_price ~ .,
                  data = df.train %>% select(one_of(c(feat_top_n, "sale_price")))) 


## create workflow 
rf3_wflow <- workflow() %>%
  add_recipe(rf3_rec) %>%
  add_model(rf3_mod)


## hyperpar. tuning

### extract transformed training set
### - we need to extract training set with only kept features
### - we use it in hyper parameters finalize step
df.train.processed <- rf3_rec %>% 
  prep() %>% 
  juice()

### extract and finalize the parameter set
### - we need to finalize the parameter set (mtry parameter)
par3_final <- extract_parameter_set_dials(rf3_mod) %>%
  finalize(df.train.processed)

rf3_tune_rez <- tune_bayes(
  rf3_wflow,
  resamples = cv_folds,
  metrics = metric_set(rmse),
  initial = 5,                              # initial grid points
  iter = 20,                                # number of optimization iterations
  param_info = par3_final,                  # hyperpar. to tune
  control = control_bayes(verbose = TRUE,
                          parallel_over = "everything")
)


## select best model 
## - model with lowest RMSE
## - and selected value for hyperpar.
rf3_mod_best <- select_best(rf3_tune_rez, metric = "rmse")


## finalize model
## - finalize workflow with best model
## - train model (model fit) on whole train data
rf3_wflow_fin <- finalize_workflow(rf3_wflow, rf3_mod_best)
rf3_mod_fit <- fit(rf3_wflow_fin, df.train)



# Model Testing / Validation

## evaluate all 3 models' performance
## - on test data (RMSE metrics)
## - first predict output on test data
## - then extract RMSE

### predict and prepare the prediction data frames
mod_fit_list <- list(mod1_all_f = rf1_mod_fit,
                     mod2_stats_remov_f = rf2_mod_fit,
                     mod3_top_n_impor_f = rf3_mod_fit)

df.test_pred <- map(mod_fit_list, ~ predict(.x, df.test) %>% 
                      bind_cols(df.test) %>% 
                      select(sale_price, .pred))

### calculate RMSE for each model
test_rmse_list <- map(df.test_pred, 
                      ~ metrics(.x, 
                                truth = sale_price, 
                                estimate = .pred) %>% 
                        filter(.metric == "rmse"))

## visualize RMSE
## - compare test RMSE for all three models
## - use test data RMSE values
bind_rows(test_rmse_list, .id = "tune") %>% 
  ggplot(aes(x = tune,
             y = .estimate,
             fill = tune,
             label = round(.estimate,1))) +
  geom_col(color = "black",
           show.legend = F) +
  geom_text(size = 10) +
  labs(title = "Random forest models: RMSE test set",
       x = "Model",
       y = "RMSE") +
  theme_minimal(base_size = 16)


## Visualize tuning results - best model
## - draw tuning hyperparameter results
## - each hyperparameter placed in its own facet
## - use facet grid to create facets for each hyperparameter 
## - each hyperparameter value VS RMSE
## - use tuning results (during training - CV)
rf1_tune_rez %>% 
  collect_metrics() %>% 
  filter(.metric == "rmse") %>% 
  select(mtry,
         min_n,
         trees,
         mean, 
         std_err) %>% 
  pivot_longer(cols = mtry:trees,
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
  labs(title = "Random forest tuning: RMSE vs. hyperparameter values",
       x = "Hyperparameter value",
       y = "RMSE") +
  theme_minimal(base_size = 16)

### selected values for hyperparam.
rf1_mod_fit$fit$actions$model$spec
