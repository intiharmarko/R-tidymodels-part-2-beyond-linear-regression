# 2 KNN and Decision Trees

rm(list = ls())
graphics.off()

# Install packages
#install.packages("kknn")
#install.packages("rpart")
#install.packages("rpart.plot")

# Load packages
library(tidyverse)
library(tidymodels)
library(kknn)
library(future)
library(doFuture)
library(rpart)
library(rpart.plot)



# 2.5 KNN with tidymodels


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
## - KNN algorithm
## - we tune (hyperpar.):
##   - neighbors ~ K (number of neighbors)
knn_mod <- nearest_neighbor(mode = "regression",
                            neighbors = tune(),
                            weight_func = "rectangular") %>%
  set_engine("kknn")

## define recipe
## - we predict log price & remove some features that are used with log version
## - first we normalize all numeric features
## - then we apply dummy encoding on category predictors
knn_rec <- recipe(formula = price_log ~ ., 
                  data = df.train %>% select(-c("price", "carat", "volume"))) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())

## create workflow 
knn_wflow <- workflow() %>%
  add_recipe(knn_rec) %>%
  add_model(knn_mod)

## set up CV for hyperpar. tuning
## - we will use k=5 fold CV
cv_folds <- vfold_cv(df.train, v = 5)

## prepare grid for tuning
## - we will use regular grid
## - n=30 different values for K
## - range between: 1 & 75
## - evenly spaced
knn_grid_regular <- grid_regular(
  neighbors(range = c(1,75)),
            levels = 30)

## hyperpar. tuning
## - execute tuning using CV data split & grid
## - we will tune using parallel computing on multiple CPU-s
## - we must register backend for foreach  
## - we must specify cores for parallel computing

registerDoFuture()                                        # parallel backend for foreach
plan(multisession, workers = parallel::detectCores() - 1) # use multi-cores
future::nbrOfWorkers()                                    # check how many workers are active

set.seed(235)

knn_tune_rez <- tune_grid(knn_wflow,
                          resamples = cv_folds,
                          grid = knn_grid_regular,
                          metrics = metric_set(rmse),
                          control = control_grid(verbose = TRUE,
                                                 parallel_over = "everything"
                                                 )
                          )


## select best model 
## - model with lowest RMSE
## - and selected value for neighbors hyperpar.
knn_mod_best <- select_best(knn_tune_rez, metric = "rmse")

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
  select(price_log, .pred)

test_rmse_knn <- df.test_knn_pred %>% 
  metrics(truth = price_log, 
          estimate = .pred) %>% 
  filter(.metric == "rmse")



# 2.9 Decision Trees with tidymodels


# Model Training

## define model specification
## - "rpart" algorithm
## - we tune (hyperpar.):
##   - min_n ~ min number of samples per leaf 
##   - tree_depth ~ max number of tree's splitting levels
##   - cost_complexity ~ alpha parameter that controls tree complexity via pruning
tree_mod <- decision_tree(mode = "regression",
                          min_n = tune(),
                          tree_depth = tune(),
                          cost_complexity = tune()) %>%
  set_engine("rpart")

## define recipe
## - we predict log price & remove some features that are used with log version
## - for decision trees we don't have to normalize all numeric features
## - also we don't have to apply dummy encoding on category predictors
tree_rec <- recipe(formula = price_log ~ ., 
                   data = df.train %>% select(-c("price", "carat", "volume"))) 

## create workflow 
tree_wflow <- workflow() %>%
  add_recipe(tree_rec) %>%
  add_model(tree_mod)

## set up CV for hyperpar. tuning
## - we will use k=10 fold CV
cv_folds <- vfold_cv(df.train, v = 10)

## prepare grid for tuning
## - we will use regular grid
## - n=30 different values for 
##
## hyperpars. ranges:
## - min_n:
##    - small data sets (< 500 rows): 2 to 20
##    - medium data sets (500-5000 rows): 5 to 30
##    - large data sets (> 5000 rows): 10 to 100
## - tree_depth :
##    - generally safe range: 2 to 10
##    - very simple (interpretable) trees: 2 to 5
##    - flexible, complex scenarios: 5 to 15
## - cost_complexity (logarithmic scale):
##    - 0.0001 to 0.1 (log10 scale: -4 to -1)
##    - most scenarios: 0.001 to 0.05
tree_grid_regular <- grid_regular(
  min_n(range = c(10, 50)),
  tree_depth(range = c(2, 5)),
  cost_complexity(range = c(-3, -1)), # translates from 0.0001 to 0.1
  levels = 5)

## hyperpar. tuning
## - execute tuning using CV data split & grid
## - we will tune using parallel computing on multiple CPU-s
## - we must register backend for foreach  
## - we must specify cores for parallel computing

registerDoFuture()                                        # parallel backend for foreach
plan(multisession, workers = parallel::detectCores() - 1) # use multi-cores
future::nbrOfWorkers()                                    # check how many workers are active

set.seed(235)

tree_tune_rez <- tune_grid(tree_wflow,
                          resamples = cv_folds,
                          grid = tree_grid_regular,
                          metrics = metric_set(rmse),
                          control = control_grid(verbose = TRUE,
                                                 parallel_over = "everything")
                          )


## select best model 
## - model with lowest RMSE
## - and selected value for hyperpar.
tree_mod_best <- select_best(tree_tune_rez, metric = "rmse")

## finalize model
## - finalize workflow with best model
## - train model (model fit) on whole train data
tree_wflow_fin <- finalize_workflow(tree_wflow, tree_mod_best)
tree_mod_fit <- fit(tree_wflow_fin, df.train)

## visualize final tree
final_tree_fit <- extract_fit_parsnip(tree_mod_fit)$fit
rpart.plot(final_tree_fit,         # your fitted rpart model
           type = 3,               # draw split labels and class probabilities (or mean if regression) *at the nodes*
           fallen.leaves = TRUE,   # leaves are placed at the bottom
           digits = 3,             # number of digits to display (3 is enough for clarity)
           roundint = FALSE,       # do not round numbers to integers (better for continuous predictions)
           box.palette = "RdYlGn", # nice color gradient from red to green (works for both regression and classification)
           shadow.col = "gray",    # subtle shadow under boxes for better contrast
           branch.lty = 1,         # solid branches (line type = 1), standard choice
           extra = 101,            # show extra information at nodes (predicted value + percentage)
           cex = 0.7               # text shrinkage if tree is large
           )



# Model Testing / Validation

## evaluate model performance
## - on test data (RMSE metrics)
## - first predict output on test data
## - then extract RMSE
df.test_tree_pred <- predict(tree_mod_fit, df.test) %>% 
  bind_cols(df.test) %>% 
  select(price_log, .pred)

test_rmse_tree <- df.test_tree_pred %>% 
  metrics(truth = price_log, 
          estimate = .pred) %>% 
  filter(.metric == "rmse")
