# Exercise - KNN for housing price predictions

rm(list = ls())
graphics.off()

# Load packages
library(tidyverse)
library(tidymodels)
library(kknn)
library(future)
library(doFuture)
library(ggplot2)


# Data

## housing prices data
load("./data/housing.RData")

## split data
set.seed(1123)

split <- initial_split(df, prop = 0.8)
df.train <- training(split)
df.test  <- testing(split)



# Model Training

## define model specification
## - KNN algorithm
## - we tune (hyperpar.):
## - neighbors ~ K (number of neighbors)
knn_mod <- nearest_neighbor(mode = "regression",
                            neighbors = tune(),
                            weight_func = "rectangular") %>%
  set_engine("kknn")

## define recipe
## - we predict median housing value
## - first we normalize all numeric features
## - then we apply dummy encoding on category predictors
knn_rec <- recipe(formula = median_house_value ~ ., 
                  data = df.train) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())

## create workflow 
knn_wflow <- workflow() %>%
  add_recipe(knn_rec) %>%
  add_model(knn_mod)

## set up CV for hyperpar. tuning
## - we will use k=5 fold CV
cv_folds <- vfold_cv(df.train, v = 5)

## hyperpar. tuning
## - execute tuning using CV data
## - tuning is executed using Bayesian optimization iterative approach 
## - we will tune using parallel computing on multiple CPU-s
## - we must register backend for foreach  
## - we must specify cores for parallel computing

registerDoFuture()                                        # parallel backend for foreach
plan(multisession, workers = parallel::detectCores() - 1) # use multi-cores
future::nbrOfWorkers()                                    # check how many workers are active

set.seed(235)

knn_bayes_rez <- tune_bayes(
  knn_wflow,
  resamples = cv_folds,
  metrics = metric_set(rmse),
  initial = 5,                                       # initial grid points
  iter = 10,                                         # number of optimization iterations
  param_info = extract_parameter_set_dials(knn_mod), # hyperpars to tune
  control = control_bayes(verbose = TRUE,
                          parallel_over = "everything")
)

## select best model 
## - model with lowest RMSE
## - and selected value for neighbors hyperpar.
knn_mod_best <- select_best(knn_bayes_rez, metric = "rmse")

## finalize model
## - finalize workflow with best model
## - train model (model fit) on whole train data
knn_wflow_fin <- finalize_workflow(knn_wflow, knn_mod_best)
knn_mod_fit <- fit(knn_wflow_fin, df.train)



# Model Testing / Validation

## evaluate model performance
## - on test data (RMSE metrics)
## - first predict output on test data
## - then extract RMSE
df.test_knn_pred <- predict(knn_mod_fit, df.test) %>% 
  bind_cols(df.test) %>% 
  select(median_house_value, .pred)

test_rmse <- df.test_knn_pred %>% 
  metrics(truth = median_house_value, 
          estimate = .pred) %>% 
  filter(.metric == "rmse")

## visualize tuning results
## - draw tuning hyperparameter results
## - hyperparameter K VS RMSE
knn_bayes_rez %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  ggplot(aes(x = neighbors, 
             y = mean)) +
  geom_point(size = 4, 
             color = "brown3") +
  geom_line(color = "gray20", 
            linewidth = 1) +
  geom_errorbar(aes(ymin = mean - std_err, 
                    ymax = mean + std_err), 
                width = 0.5, 
                alpha = 0.5) +
  scale_x_continuous(breaks = seq(1,25,1)) +
  labs(title = "KNN Tuning: RMSE vs. Number of Neighbors",
       x = "Neighbors (K)",
       y = "RMSE") +
  theme_minimal(base_size = 16)
