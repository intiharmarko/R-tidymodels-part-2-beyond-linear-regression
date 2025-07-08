# 5 Final Project: Japan Real Estate Prices Prediction - Modeling - Functions


#' Split training data into train and validate sets
#'
#' This function splits the data based on time split into:
#' - train set
#' - validate set
#' Split is done using pivot date, based on which the function cuts the data 
#' into validation set and training set. Data with date greater or equal to pivot date
#' is used in validate set, data with date less than pivot date is used for training set.
#' You need to provide a training data frame and pivot date.
#' 
#' @param data A data frame for selected data.
#' @param pivot_date A date used for split the data.
#' @param var_date A string indicating the name of date like column we used for data split.
#'
#' @return data_split.list A list of data frame split into sets (training & validate).
#'
split_data_date <- function(data,
                            pivot_date,
                            var_date = "date"){
  
  # sort data (time ascending order)
  data <- data %>% 
    arrange(.data[[var_date]])
  
  # split data
  df.training <- data %>% filter(.data[[var_date]] < pivot_date)
  df.validate <- data %>% filter(.data[[var_date]] >= pivot_date)
  
  # print data set dimension sizes
  print("Data split into:")
  print(paste0("train set: ",    nrow(df.training),    " rows | ", round(nrow(df.training) / nrow(data) * 100, 2),    
               " % of all rows | date span: ", df.training[[var_date]] %>% min(), " ~ ", df.training[[var_date]] %>% max()))
  print(paste0("validate set: ", nrow(df.validate), " rows | ", round(nrow(df.validate) / nrow(data) * 100, 2), 
               " % of all rows | date span: ", df.validate[[var_date]] %>% min(), " ~ ", df.validate[[var_date]] %>% max()))
  
  # create list
  data_split.list <- list(training = df.training,
                          validate =  df.validate)
  
  return(data_split.list)
}


#' Finalize model parameter list
#'
#' This function matches selected recipe with selected model and finalizes features (parameter) list.
#' For the hyperparameters tuning phase.
#' 
#' @param mod_list_ A list of model specification.
#' @param rec_list_ A list of recipes used for each model.
#' @param df.train_ A training data df.
#'
#' @return par_fin_list A list of finalized features per each model.
#'
finalize_model_params <- function(mod_list_ = mod_list,
                                  rec_list_ = rec_list,
                                  df.train_ = df.training){
  
  # finalize model's parameter list (matching model and recipe)
  par_fin_list <- map(names(mod_list_), function(id) {
    
    # first prep and bake the corresponding recipe
    rec <- prep(rec_list_[[id]], training = df.train_)
    df.train_baked <- bake(rec, new_data = NULL)
    
    # extract and finalize model parameters using baked data
    extract_parameter_set_dials(mod_list[[id]]) %>%
      finalize(df.train_baked)
  })
  
  # assign model's names to list of results
  names(par_fin_list) <- names(mod_list)
  
  return(par_fin_list)
}


#' Tune hyperparameters of multiple algorithms with Bayes optimization tuning algorithm
#' 
#' This functions runs tuning hyperparameters of selected models (different algorithms)
#' using Bayesian optimization tuning algorithm. New model candidates are generated
#' based on tuning parameter combinations from previous results (tuning iterations).
#' It is custom made function based on tune_bayes(), which can be applied for multiple
#' modeling paradigms - algorithm types in single run.
#' 
#' @param wflow_set_ A tible of workflows objects per each model.
#' @param par_fin_list_ A list of finalized features per each model.
#' @param cv_folds_ Train data split into folds (Cross-validation object) / also supports date based CV objects.
#' @param iter The maximum number of search iterations.
#' @param initial An initial set of results.
#' @param no_improve The integer cutoff for the number of iterations without better results.
#' 
#' @return tune_rez A list of tuning results per each model. 
#'
tune_bayes_custom <- function(wflow_set_ = wflow_set,
                              par_fin_list_ = par_fin_list,
                              cv_folds_,
                              iter = 20,
                              initial = 5,
                              no_improve = 5){
  
  # empty list - tuning results
  tune_rez <- list()
  
  # loop over each model id
  for (id in wflow_set_$wflow_id) {
    
    print("")
    print("|--------------------------|")
    print(paste0("Tuning: ", id))
    print(paste0("Started at: ", Sys.time()))
    
    ts <- Sys.time() # track execution time
    
    # extract workflow
    wflow <- extract_workflow(wflow_set_, id)
    
    # extract finalized param set
    par_fin <- par_fin_list_[[id]]
    
    # run Bayesian tuning
    rez <- tune_bayes(
      object = wflow,
      resamples = cv_folds_,
      param_info = par_fin,
      iter = iter,
      initial = initial,
      control = control_bayes(verbose = TRUE, 
                              parallel_over = "everything",
                              no_improve = no_improve, 
                              save_pred = TRUE)
    )
    
    # report execution time
    te <- Sys.time()
    t_el <- round(as.numeric(difftime(te, ts, units = "mins")), 2)
    print(paste0("Finished tuning for: ", id, " in ", t_el, " minutes."))
    print(paste0("Ended at: ", Sys.time()))
    
    # store results
    tune_rez[[id]] <- rez
  }
  
  return(tune_rez)
}


#' Finalize each workflow and fit best model.
#' 
#' This functions first selects best model candidate per each model (algorithm type),
#' and then fits best model candidate using complete training data, 
#' 
#' @param tune_rez_ A list of tuning results per each model. 
#' @param wflow_set_ A tibble of workflows objects per each model.
#' @param df.train_ A training data df.
#' 
#' @return mod_fit_list A list of fitted models (best candidates per algorithm) on training data.
#'
fit_best_models <- function(tune_rez_ = tune_rez,
                            wflow_set_ = wflow_set,
                            df.train_ = df.training){
  
  mod_fit_list  <- map(names(tune_rez_), function(id) {
    
    # extract best model candidate per given algorithm
    best_par <- select_best(tune_rez_[[id]], metric = "rmse")
    
    # fit final model
    extract_workflow(wflow_set_, id) %>%
      finalize_workflow(best_par) %>% 
      fit(data = df.train_)
  })
  
  # assign model's names to list of results
  names(mod_fit_list) <- names(tune_rez_)
  
  return(mod_fit_list)
}


#' Save R's work space snapshot
#' 
#' This functions stores current R's work space on PC disk. 
#' 
#' @param path A string containing path to .RData file on disk.
#' 
#' @return NULL
#'
save_WS_snapshot <- function(path){
  
  save.image(file = path)
  
  cat("\nR's work space image created and stored to:")
  cat("\n", path)
}


#' Load stored R's work space snapshot
#' 
#' This functions loads stored R's work space from disk to R session. 
#' 
#' @param path A string containing path to .RData file on disk.
#' 
#' @return NULL
#'
load_WS_snapshot <- function(path){
  
  load(file = path, envir = .GlobalEnv)
  
  cat("\nR's work space image loaded from:")
  cat("\n", path)
}


#' Get full algorithm name for best selected model
#' 
#' This functions will return full algorithm name based on abbreviated algorithm name. 
#' 
#' @param alg_best_ A string holding abbreviated algorithm name, which was selected.
#' 
#' @return alg_best_name A string representing full algorithm name.
#'
get_alg_full_name <- function(alg_best_ = alg_best){
  
  if(alg_best == "xgb"){
    
    alg_best_name <- "XGBoost"
    
  } else if(alg_best == "lgbm"){
    
    alg_best_name <- "LigthGBM"
    
  } else{
    
    alg_best_name <- alg_best
    
  }
  
  return(alg_best_name)
}


#' Fit final best single model.
#' 
#' This functions first selects best model candidate per best model (algorithm type),
#' and then fits best model candidate using complete training data, 
#' 
#' @param alg_best_ A string holding abbreviated algorithm name, which was selected.
#' @param tune_rez_ A list of tuning results per each model. 
#' @param wflow_set_ A tibble of workflows objects per each model.
#' @param df.train_ A training data df.
#' 
#' @return mod_fit_list A list of fitted models (best candidates per algorithm) on training data.
#'
fit_fin_model <- function(alg_best_ = alg_best,
                          tune_rez_ = tune_rez,
                          wflow_set_ = wflow_set,
                          df.train_ = df.train){
  
 
  # extract best model candidate per given algorithm
  best_par <- select_best(tune_rez_[[alg_best_]], metric = "rmse")
    
  ## finalize model
  ## - finalize workflow with best model
  ## - train model (model fit) on whole train data
  mod_fit <- extract_workflow(wflow_set_, alg_best_) %>% 
    finalize_workflow(best_par) %>% 
    fit(data = df.train_)
    
  return(mod_fit)
}
