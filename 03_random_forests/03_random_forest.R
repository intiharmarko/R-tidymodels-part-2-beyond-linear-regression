# 3 Random Forest

rm(list = ls())
graphics.off()

# Install packages
#install.packages("ranger")

# Load packages
library(tidyverse)
library(tidymodels)
library(ranger)
library(future)
library(doFuture)



# 3.4 Random forest with tidymodels


# Data

## diamonds data set
load("./data/diamonds.RData")

## split.data
set.seed(1123)

split <- initial_split(df.m, prop = 0.8) # we use medium size diamonds df!
df.train <- training(split)
df.test  <- testing(split)


# Model Training

## define model specification
## - "ranger" algorithm
## - we tune (hyperpar.):
##   - mtry ~ number of predictors (features) sampled at each split
##   - min_n ~ min number of observations per node
##   - trees ~ number of trees in the forest
##
## hyperpars. ranges:
## - mtry:
##    - in regression default value - (number of features) / 3
##    - used range: (number of features / 3) to (number of features)
## - min_n:
##    - safe range: 2 to 10 (works well for most problems) 
## - trees:
##    - often fixed at 1000
##    - used range: 500 to 2000
##    - tuning trees rarely improves performance significantly!

rf_mod <- rand_forest(mode = "regression",
                      mtry = tune(),
                      min_n = tune(),
                      trees = tune()) %>%
  set_engine("ranger")

## define recipe
## - we predict log price & remove some features that are used with log version
## - for random forest alg. we don't have to normalize all numeric features
## - also we don't have to apply dummy encoding on category predictors
rf_rec <- recipe(formula = price_log ~ .,
                 data = df.train %>% select(-c("price", "carat", "volume"))) 

## create workflow 
rf_wflow <- workflow() %>%
  add_recipe(rf_rec) %>%
  add_model(rf_mod)

## set up CV for hyperpar. tuning
## - we will use k=10 fold CV
cv_folds <- vfold_cv(df.train, v = 10)

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

### extract and finalize the parameter set
### - we need to finalize the parameter set, 
### - so that mtry knows the valid range (e.g., 1 to number of predictors in your data)
### - without parameter finalization part - we get error during tuning (mtry hyperpar. causes the error!)
par_final <- extract_parameter_set_dials(rf_mod) %>%
  finalize(select(df.train, -c("price", "carat", "volume")))

rf_tune_rez <- tune_bayes(
  rf_wflow,
  resamples = cv_folds,
  metrics = metric_set(rmse),
  initial = 5,                              # initial grid points
  iter = 20,                                # number of optimization iterations
  param_info = par_final,                    # hyperpar. to tune
  control = control_bayes(verbose = TRUE,
                          parallel_over = "everything")
)


## select best model 
## - model with lowest RMSE
## - and selected value for hyperpar.
rf_mod_best <- select_best(rf_tune_rez, metric = "rmse")

## finalize model
## - finalize workflow with best model
## - train model (model fit) on whole train data
rf_wflow_fin <- finalize_workflow(rf_wflow, rf_mod_best)
rf_mod_fit <- fit(rf_wflow_fin, df.train)


# Visualize training results 

## Visualize tuning results
## - draw tuning hyperparameter results
## - each hyperparameter placed in its own facet
## - use facet grid to create facets for each hyperparameter 
## - each hyperparameter value VS RMSE
## - use tuning results (during training - CV)
rf_tune_rez %>% 
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
rf_mod_fit$fit$actions$model$spec



# Model Testing / Validation

## evaluate model performance
## - on test data (RMSE metrics)
## - first predict output on test data
## - then extract RMSE
df.test_rf_pred <- predict(rf_mod_fit, df.test) %>% 
  bind_cols(df.test) %>% 
  select(price_log, .pred)

test_rmse <- df.test_rf_pred %>% 
  metrics(truth = price_log, 
          estimate = .pred) %>% 
  filter(.metric == "rmse")

