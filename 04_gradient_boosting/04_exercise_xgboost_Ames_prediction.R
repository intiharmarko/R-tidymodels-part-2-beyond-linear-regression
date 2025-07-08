# Exercise - XGBoost for Ames data sale price prediction

rm(list = ls())
graphics.off()


# Load packages
library(tidyverse)
library(tidymodels)
library(xgboost)
library(future)
library(doFuture)
library(janitor)



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



# Model Training

## define model specification
## - "xgboost" algorithm
## - we tune (hyperpar.):
##   - trees ~ number of trees (learners)
##   - tree_depth ~ max depth of each tree
##   - learn_rate ~ learning rate in boosting step

xgb_mod <- boost_tree(mode = "regression",
                      trees = tune(),
                      tree_depth = tune(),
                      learn_rate = tune()) %>%
  set_engine("xgboost")

## define recipe
## - we predict log price & remove some features that are used with log version
## - for XGBoost alg. we have to dummy encode categorical features
xgb_rec <- recipe(formula = sale_price ~ .,
                 data = df.train) %>% 
  step_dummy(all_nominal_predictors())

## create workflow 
xgb_wflow <- workflow() %>%
  add_recipe(xgb_rec) %>%
  add_model(xgb_mod)

## hyperpar. tuning
## - execute tuning using CV data
## - tuning is executed using Bayesian optimization 
## - Bayes optimization tuning specs:
##   - 5 initial points 
##   - max 20 iterations
## - we will tune using parallel computing on multiple CPU-s
## - we must register backend for foreach  
## - we must specify cores for parallel computing

registerDoFuture()                                        # parallel backend for foreach
plan(multisession, workers = parallel::detectCores() - 1) # use multi-cores
future::nbrOfWorkers()                                    # check how many workers are active

set.seed(235)

xgb_tune_rez <- tune_bayes(
  xgb_wflow,
  resamples = cv_folds,
  metrics = metric_set(rmse),
  initial = 5,                                        # initial grid points
  iter = 1,                                          # number of optimization iterations
  param_info = extract_parameter_set_dials(xgb_mod),  # hyperpar. to tune
  control = control_bayes(verbose = TRUE,
                          parallel_over = "everything")
)


## select best model 
## - model with lowest RMSE
## - and selected value for hyperpar.
xgb_mod_best <- select_best(xgb_tune_rez, metric = "rmse")

## finalize model
## - finalize workflow with best model
## - train model (model fit) on whole train data
xgb_wflow_fin <- finalize_workflow(xgb_wflow, xgb_mod_best)
xgb_mod_fit <- fit(xgb_wflow_fin, df.train)


# Visualize training results 

## Visualize tuning results
## - draw tuning hyperparameter results
## - each hyperparameter placed in its own facet
## - use facet grid to create facets for each hyperparameter 
## - each hyperparameter value VS RMSE
## - use tuning results (during training - CV)
xgb_tune_rez %>% 
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
  labs(title = "XGBoost tuning: RMSE vs. hyperparameter values",
       x = "Hyperparameter value",
       y = "RMSE") +
  theme_minimal(base_size = 16)

### selected values for hyperparam.
xgb_mod_fit$fit$actions$model$spec



# Model Testing / Validation

## evaluate model performance
## - on test data (RMSE metrics)
## - first predict output on test data
## - then extract RMSE
df.test_xgb_pred <- predict(xgb_mod_fit, df.test) %>% 
  bind_cols(df.test) %>% 
  select(sale_price, .pred)

test_rmse <- df.test_xgb_pred %>% 
  metrics(truth = sale_price, 
          estimate = .pred) %>% 
  filter(.metric == "rmse")
