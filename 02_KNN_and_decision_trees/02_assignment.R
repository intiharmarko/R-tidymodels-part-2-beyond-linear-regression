# Assignment - Decision Trees for housing price predictions

rm(list = ls())
graphics.off()

# Load packages
library(tidyverse)
library(tidymodels)
library(future)
library(doFuture)
library(rpart)
library(finetune)


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
## - "rpart" algorithm
## - we tune hyperparameters
tree_mod <- decision_tree(mode = "regression",
                          min_n = tune(),
                          tree_depth = tune(),
                          cost_complexity = tune()) %>%
  set_engine("rpart")

## define recipe
## - we predict median housing value
tree_rec <- recipe(formula = median_house_value ~ ., 
                   data = df.train) 

## create workflow 
tree_wflow <- workflow() %>%
  add_recipe(tree_rec) %>%
  add_model(tree_mod)

## set up CV for hyperpar. tuning
## - we will use k=10 fold CV
cv_folds <- vfold_cv(df.train, v = 10)

## hyperpar. tuning
## - execute tuning using CV data
## - tuning is executed using two iterative approaches:
##   - Bayesian optimization 
##   - Simulated annealing
## - Bayes optimization tuning specs:
##   - 20 initial points & max 100 iterations
## - Simulated annealing tuning specs:
##   - 20 initial points & max 100 iterations
##   - control: restart = 5 | radius = c(0.05, 0.5) | flip = 0.05 | cooling_coef = 0.1
## - we will tune using parallel computing on multiple CPU-s
## - we must register backend for foreach  
## - we must specify cores for parallel computing

registerDoFuture()                                        # parallel backend for foreach
plan(multisession, workers = parallel::detectCores() - 1) # use multi-cores
future::nbrOfWorkers()                                    # check how many workers are active

set.seed(235)

tree_bayes_rez <- tune_bayes(
  tree_wflow,
  resamples = cv_folds,
  metrics = metric_set(rmse),
  initial = 20,                                         # initial grid points
  iter = 100,                                           # number of optimization iterations
  param_info = extract_parameter_set_dials(tree_mod),   # hyperpar. to tune
  control = control_bayes(verbose = TRUE,
                          parallel_over = "everything")
)

tree_siann_rez <- tune_sim_anneal(
  tree_wflow,
  resamples = cv_folds,
  metrics = metric_set(rmse),
  initial = 20,                                        # initial grid points
  iter = 100,                                          # number of optimization iterations
  param_info = extract_parameter_set_dials(tree_mod),  # hyperpar. to tune
  control = control_sim_anneal(
    verbose = TRUE,               # print progress
    parallel_over = "everything", # parallel across everything
    restart = 5,                  # allow up to 5 random restarts
    radius = c(0.05, 0.5),        # search step size range (small to moderate jumps)
    flip = 0.05,                  # 5% chance to move completely randomly
    cooling_coef = 0.1,           # cooling speed (higher = faster cooling)
    save_history = TRUE           # keep track of all tried points
  )
)

## select best model & finalize it
## - model with lowest RMSE
## - and selected values for hyperpar.
## - finalize workflow with best model
## - train model (model fit) on whole train data
## - for both tuning techniques
tree_mod_fit_list <- list(bayes = tree_bayes_rez,
                          siann = tree_siann_rez) %>% 
  map(select_best, metric = "rmse") %>% 
  map(~ finalize_workflow(tree_wflow, .x)) %>% 
  map(~ fit(.x, df.train))



# Model Testing / Validation

## evaluate models performance
## - on test data (RMSE metrics)
## - first predict output on test data
## - then extract RMSE

### predict and prepare the prediction data frames
df.test_pred <- map(tree_mod_fit_list, ~ predict(.x, df.test) %>% 
                      bind_cols(df.test) %>% 
                      select(median_house_value, .pred))

### calculate RMSE for each model
test_rmse_list <- map(df.test_pred, 
                      ~ metrics(.x, 
                                truth = median_house_value, 
                                estimate = .pred) %>% 
                        filter(.metric == "rmse"))



# Visualize results 

## Visualize tuning results
## - draw tuning hyperparameter results
## - each hyperparameter placed in its own facet
## - also split plot based on tuning strategy
## - use facet grid to create facets for each hyperparameter & each tuning strategy 
## - each hyperparameter value VS RMSE
## - use tuning results (during training - CV)
list(bayes = tree_bayes_rez,
     siann = tree_siann_rez) %>% 
  map(collect_metrics) %>% 
  map(filter, .metric == "rmse") %>% 
  bind_rows(., .id = "tune") %>% 
  select(tune, 
         min_n, 
         tree_depth, 
         cost_complexity, 
         mean, 
         std_err) %>% 
  pivot_longer(cols = min_n:cost_complexity,
               names_to = "par",
               values_to = "par_value") %>% 
  ggplot(aes(x = par_value, 
             y = mean,
             group = tune)) +
  geom_point(size = 4, 
             color = "brown3") +
  geom_line(color = "gray20", 
            linewidth = 1) +
  geom_errorbar(aes(ymin = mean - std_err, 
                    ymax = mean + std_err), 
                width = 0.5, 
                alpha = 0.5) +
  facet_grid(rows = vars(tune),
             cols = vars(par),
             scales = "free",
             labeller = "label_both") +
  labs(title = "Decision trees tuning: RMSE vs. hyperparameter values",
       x = "Hyperparameter value",
       y = "RMSE") +
  theme_minimal(base_size = 16)

### selected values for hyperparam.
tree_mod_fit_list$bayes$fit$actions$model$spec
tree_mod_fit_list$siann$fit$actions$model$spec

## visualize actual prices VS predicted prices
## - draw actual housing prices VS predicted prices
## - draw each point (actual & predicted price)
## - consider both tuning strategies
## - use test data
bind_rows(df.test_pred, .id = "tune") %>% 
  ggplot(aes(x = median_house_value,
             y = .pred,
             color = tune)) +
  geom_jitter(alpha = 0.4,
              size = 3) +
  scale_color_viridis_d(option = "magma") +
  labs(title = "Decision trees tuning: actual price vs. predicted price",
       x = "Actual median house value",
       y = "Predicted median house value",
       color = "Tune technique:") +
  theme_minimal(base_size = 16)

## visualize RMSE
## - compare test RMSE for both models (different tuning techniques)
## - use test data RMSE values
bind_rows(test_rmse_list, .id = "tune") %>% 
  ggplot(aes(x = tune,
             y = .estimate,
             fill = tune,
             label = round(.estimate,1))) +
  geom_col(color = "black",
           show.legend = F) +
  geom_text(size = 10) +
  labs(title = "Decision trees tuning: RMSE test set",
       x = "Tune technique",
       y = "RMSE") +
  theme_minimal(base_size = 16)
