# Assignment - Selected ML algorithms for Ames data sale price prediction - Functions


#' Split data
#'
#' This function randomly splits the data into:
#' - train set
#' - validate set
#' - test set
#' You need to provide a data frame and set's proportions.
#' 
#' @param df A data frame for selected data.
#' @param p_t A numerical value - proportion of data assigned to train set (between 0 and 1 - must be less than 1!) - (default value 7/10).
#' @param p_v A numerical value - proportion of test set data assigned to validate set (between 0 and 1 - must be greater than 0!) - (default value 2/3).
#'
#' @return data_split.list A list of data frame split into sets.
#'
split_data <- function(df,
                       p_t = 7/10,
                       p_v = 2/3){
  
  # train VS validate + test split
  split_init  <- initial_split(df, prop = p_t) # initial split train VS validate + test
  df.train    <- training(split_init)          # train data
  df.val_test <- testing(split_init)           # validate + test data
  
  # validate VS test split
  split_val_tes <- initial_split(df.val_test, prop = p_v) # split validate VS test
  df.validate   <- training(split_val_tes)                # validate data
  df.test       <- testing(split_val_tes)                 # test data
  
  # print data set dimension sizes
  print("Data split into:")
  print(paste0("train set: ",    nrow(df.train),    " rows | ", round(nrow(df.train) / nrow(df) * 100, 2),    " % of all rows"))
  print(paste0("validate set: ", nrow(df.validate), " rows | ", round(nrow(df.validate) / nrow(df) * 100, 2), " % of all rows"))
  print(paste0("test set: ",     nrow(df.test),     " rows | ", round(nrow(df.test) / nrow(df) * 100, 2),     " % of all rows"))

  # create list
  data_split.list <- list(train = df.train,
                          validate =  df.validate,
                          test = df.test)
  
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
                                  df.train_ = df.train){
  
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
#' @param cv_folds_ Train data split into folds (Cross-validation object).
#' @param iter The maximum number of search iterations.
#' @param initial An initial set of results.
#' @param no_improve The integer cutoff for the number of iterations without better results.
#' 
#' @return tune_rez A list of tuning results per each model. 
#'
tune_bayes_custom <- function(wflow_set_ = wflow_set,
                              par_fin_list_ = par_fin_list,
                              cv_folds_ = cv_folds,
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
                            df.train_ = df.train){
  
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
