#' 5 Final Project: Japan Real Estate Prices Prediction - EDA - Functions


#' Load packages
library(tidyverse)


#' Column types count
#'
#' This function draws a bar plot counting columns by column type, 
#' also target column is highlighted.
#' 
#' @param df.cols_ A data frame of columns with column types added.
#'
#' @return NULL
#'
col_type_count <- function(df.cols_ = df.cols){
  
  df.cols %>% 
    group_by(role, type) %>% 
    count() %>% 
    ungroup() %>% 
    ggplot(aes(x = type,
               y = n,
               fill = role)) +
    geom_col(color = "black") +
    labs(title = "Varible types count",
         fill = "Variable role:") +
    xlab("Column type") +
    ylab("Count") +
    scale_fill_manual(values = c("grey75", "brown1")) +
    theme_minimal(base_size = 16)
}


#' Missing rows % count
#'
#' This function draws a plot showing percentage of missing rows per each variable (column).
#' 
#' @param df A data frame for given data.
#'
#' @return NULL
#'
rows_miss_count <- function(df = df.train){
  
  map(.x = df,
      .f = ~sum(is.na(.))) %>% 
    unlist() %>% 
    tibble(col = names(.),
           nr_missing = .,
           per_missing = round(nr_missing / nrow(df) * 100, 2)) %>% 
    ggplot(aes(y = col,
               x = per_missing,
               color = per_missing)) +
    geom_point(size = 4) +
    scale_x_continuous(breaks = seq(0,100,10),
                       limits = c(0, 100)) +
    scale_color_viridis_c(option = "magma") +
    labs(title = "Percentage of missing rows by column",
         color = "% missing rows:") +
    xlab("Percentage of missing rows") +
    ylab("Column") +
    theme_minimal(base_size = 16)
}


#' Count data points (nr rows) per reported date (quarter)
#'
#' This function first calculates number of rows per each date (quarterly data),
#' and then draws row counts per date.
#' 
#' @param df A data frame for given data.
#'
#' @return NULL
#'
rows_date_count <- function(df = df.train){
  
  # get min / max date
  date.min <- df %>% summarise(date_min = min(date)) %>% pull(date_min)
  date.max <- df %>% summarise(date_max = max(date)) %>% pull(date_max)
  
  # draw plot
  df %>% 
    summarise(n = n(), 
              .by = date) %>% 
    ggplot(aes(x = date,
               y = n)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 4,
               color = "brown1") +
    scale_x_date(breaks = seq(date.min, date.max, by = "3 months"),
                 labels = function(x) paste0(year(x), " Q", quarter(x)),
                 limits = c(date.min, date.max)) +
    labs(title = "Number of data points (rows) per reported quaterly period") +
    xlab("Reported period (quaterly reports)") +
    ylab("Number of data points (rows)") +
    theme_minimal(base_size = 16) +
    theme(axis.text.x = element_text(angle = 90))
}


#' Count number of categories (levels) of categorical columns
#'
#' This function first converts character columns into factors, then counts number of
#' different categories of each factor, and draws a bar plot of factors and their
#' number of different categories. We can control max number of levels to be shown 
#' for factor variables on the plot.
#' 
#' @param df A data frame for given data.
#' @param n_max An integer- max number of levels / categories for factor variable to be drawn on plot.
#'
#' @return NULL
#'
nr_categories_count <- function(df = df.train,
                                n_max = nrow(df)){
  
  # convert characters to factors and count categories
  df.counts <- df.train %>% 
    mutate(across(.cols = where(is.character), 
                  .fns = as.factor)) %>% 
    select(where(is.factor)) %>% 
    map_int(., nlevels) %>%
    enframe(name = "col",
            value = "n") %>%
    arrange(n)
  
  # draw categories counts
  df.counts %>% 
    filter(n <= n_max) %>% 
    ggplot(aes(x = n,
               y = fct_inorder(col))) +
    geom_col(color = "black",
             fill = "gray80") +
    ggtitle("Factor features - unique levels count") +
    xlab("Number of unique levels (categories)") +
    ylab("Feature") +
    theme_minimal(base_size = 16)
}  


#' Target variable distribution
#'
#' This function draws distribution of "trade_price" - density plot & box plot.
#' You choose type of plot:
#'   - "dens" ~ density plot 
#'   - "box"  ~ box plot
#' 
#' @param df A data frame for given data.
#' @param p.type A string controlling type of plot: "dens" (density plot) | "box" (box plot).
#'
#' @return NULL
#'
target_var_distr <- function(df = df.train,
                             p.type = "dens"){
  
  if(p.type == "dens"){
    
    # density plot
    df %>% 
      ggplot(aes(x = trade_price)) +
      geom_density(fill = "gray90",
                   color = "black") +
      scale_x_log10() +
      ggtitle("Target varible (trade price) density plot") +
      xlab("Trade price (in yen) - log10 scale") +
      ylab("Density") +
      theme_minimal(base_size = 16)
    
  } else if(p.type == "box"){
    
    # box plot
    df %>% 
      ggplot(aes(y = trade_price)) +
      geom_boxplot(fill = "gray90",
                   color = "black") +
      scale_y_log10() +
      ggtitle("Target varible (trade price) box plot") +
      ylab("Trade price (in yen) - log10 scale") +
      theme_minimal(base_size = 16)
    
  } else{
    message("Please select: 'dens' or 'box' option!")
  }
}


#' Target variable (trade price) over time
#'
#' This function plots trade price over time - distribution.
#' You choose type of plot:
#'   - "sct"  ~ scatter plot 
#'   - "box"  ~ box plot
#' 
#' @param df A data frame for given data.
#' @param p.type A string controlling type of plot: "dens" (density plot) | "box" (box plot).
#' @param n_points An integer indicating the size of points being randomly sampled - only for scatter plot!
#'
#' @return NULL
#'
target_var_time_distr <- function(df = df.train,
                                  p.type = "sct",
                                  n_points = 50000){
  
  # get min / max date
  date.min <- df %>% summarise(date_min = min(date)) %>% pull(date_min)
  date.max <- df %>% summarise(date_max = max(date)) %>% pull(date_max)
  
  if(p.type == "sct"){
    
    # scatter plot
    df %>% 
      dplyr::sample_n(size = n_points) %>% 
      ggplot(aes(x = date,
                 y = trade_price)) +
      geom_jitter(alpha = 1/10) +
      scale_x_date(breaks = seq(date.min, date.max, by = "3 months"),
                   labels = function(x) paste0(year(x), " Q", quarter(x)),
                   limits = c(date.min, date.max)) +
      scale_y_log10() +
      labs(title = "Trade price per reported quaterly period - scatter plot") +
      xlab("Reported period (quaterly reports)") +
      ylab("Trade price (in yen) - log10 scale") +
      theme_minimal(base_size = 16) +
      theme(axis.text.x = element_text(angle = 90))
    
  } else if(p.type == "box"){
    
    # box plot
    df %>% 
      ggplot(aes(x = date,
                 y = trade_price,
                 group = date)) +
      geom_boxplot() +
      scale_x_date(breaks = seq(date.min, date.max, by = "3 months"),
                   labels = function(x) paste0(year(x), " Q", quarter(x))) +
      scale_y_log10() +
      labs(title = "Trade price per reported quaterly period - box plot") +
      xlab("Reported period (quaterly reports)") +
      ylab("Trade price (in yen) - log10 scale") +
      theme_minimal(base_size = 16) +
      theme(axis.text.x = element_text(angle = 90))
    
  } else{
    message("Please select: 'sct' or 'box' option!")
  }
}


#' Plot histogram - numeric variable distribution
#'
#' This function draws distribution of selected numeric variable using histogram.
#' 
#' @param data A data frame for given data.
#' @param x_var A string - name of variable we are plotting.
#' @param bins_ An integer - number of bins in histogram.
#'
#' @return NULL
#'
plot_histogram <- function(data = df.train, 
                           x_var, 
                           bins_ = 30) {
  
  ggplot(data, 
         aes(x = .data[[x_var]])) + 
    geom_histogram(fill = "gray90",
                   color = "black",
                   bins = bins_) +
    ggtitle(paste0("Numeric feature (", x_var, ") distribution -  histrogram")) +
    xlab("") +
    ylab("Counts") +
    theme_minimal(base_size = 16)
}


#' Plot density plot - numeric variable distribution
#'
#' This function draws distribution of selected numeric variable using density plot.
#' 
#' @param data A data frame for given data.
#' @param x_var A string - name of variable we are plotting.
#'
#' @return NULL
#'
plot_density <- function(data = df.train, 
                         x_var) {
  
  ggplot(data, 
         aes(x = .data[[x_var]])) + 
    geom_density(fill = "gray90",
                 color = "black") +
    ggtitle(paste0("Numeric feature (", x_var, ") distribution -  density plot")) +
    xlab("") +
    ylab("Density") +
    theme_minimal(base_size = 16)
}


#' Plot bar plot - factor variable distribution
#'
#' This function draws distribution of selected factor variable using bar plot.
#' 
#' @param df A data frame for given data.
#' @param x_var A string - name of variable we are plotting.
#' @param x_font_size An integer indicating the size of fonts on x axis (ticks).
#'
#' @return NULL
#'
plot_fct_dist <- function(data = df.train, 
                          x_var,
                          x_font_size = 8) {
  
  # calculate counts per each level
  plot_data <- data %>% 
    count(.data[[x_var]], 
          name = "n") %>% 
    arrange(desc(n)) %>% 
    mutate(per = n / sum(n),
           per_csum = cumsum(per))
  
  print(plot_data)
  
  # draw plot
  ggplot(plot_data,
         aes(x = fct_inorder(.data[[x_var]]), 
             y = n)) +
    geom_col(color = "black", 
             fill = "gray80") +
    labs(title = paste0("Factor feature (", x_var, ") distribution – bar plot"),
         x = NULL,
         y = "Counts") +
    theme_minimal(base_size = 16) +
    theme(axis.text.x = element_text(angle = 90, 
                                     size = x_font_size))
}


#' Plot target variable VS numeric feature distribution - scatter plot
#'
#' This function draws distribution of selected numeric feature and numeric 
#' target variable variable using scatter plot. Data points are additionally 
#' being sample (to optimize scatter plot drawing).
#' 
#' @param data A data frame for given data.
#' @param x_var A string - name of numeric feature variable we are plotting.
#' @param n_points An integer indicating the size of points being randomly sampled.
#'
#' @return NULL
#'
plot_tar_num_feat_dist <- function(data = df.train, 
                                   x_var,
                                   n_points = 50000) {
  
  data %>% 
    dplyr::sample_n(size = n_points) %>% 
    ggplot(aes(x = .data[[x_var]],
               y = trade_price)) +
    geom_jitter(alpha = 1/10) +
    scale_y_log10() +
    xlab(paste0(x_var)) +
    ylab("Trade price (in yen) - log10 scale") +
    labs(title = paste0("Target varible (trade_price) VS ", x_var, " - scatter plot"),
         subtitle = paste0(n_points, " points randomly sampled (shown on figure)")) +
    theme_minimal(base_size = 16)
  
}


#' Plot target variable VS factor feature distribution - box plot
#'
#' This function draws distribution of selected factor feature and numeric 
#' target variable variable using box plot. 
#' 
#' @param data A data frame for given data.
#' @param x_var A string - name of factor feature variable we are plotting.
#' @param x_font_size An integer indicating the size of fonts on x axis (ticks).
#'
#' @return NULL
#'
plot_tar_fct_feat_dist <- function(data = df.train, 
                                   x_var,
                                   x_font_size = 8) {
  
  data %>% 
    ggplot(aes(x = .data[[x_var]],
               y = trade_price)) +
    geom_boxplot() +
    scale_y_log10() +
    xlab(paste0(x_var)) +
    ylab("Trade price (in yen) - log10 scale") +
    ggtitle(paste0("Target varible (trade_price) VS ", x_var, " - scatter plot")) +
    theme_minimal(base_size = 16) +
    theme(axis.text.x = element_text(angle = 90, 
                                     size = x_font_size))
}  


#' Plot correlation heatmap - numeric variables
#'
#' This function firsts calculates Pearson correlation coefficient for all pairs
#' of numeric variables, and then draws heatmap.
#' 
#' @param data A data frame for given data.
#' @param df.cols_ A data frame of columns with column types added.
#'
#' @return NULL
#'
plot_corr_heatmap <- function(data = df.s.train,
                              df.cols_ = df.cols){
  
  # compute correlation matrix (Person coefficient)
  cor_matrix <- df.s.train %>%
    select(one_of(df.cols %>% 
                    filter(type == "numeric") %>% 
                    pull(col))) %>%
    cor(method = "pearson", use = "complete.obs")
  
  # convert to long format for ggplot
  cor_long <- cor_matrix %>%
    as.data.frame() %>%
    rownames_to_column("var1") %>%
    pivot_longer(-var1, names_to = "var2", values_to = "correlation")
  
  # plot the heat map
  ggplot(cor_long, 
         aes(x = var1, 
             y = var2, 
             fill = correlation)) +
    geom_tile(color = "white") +
    geom_text(aes(label = round(correlation, 2)), 
              size = 3) +
    scale_fill_gradient2(low = "blue", 
                         mid = "white", 
                         high = "red",
                         midpoint = 0, 
                         limit = c(-1, 1), 
                         name = "Pearson corr.:") +
    xlab("") +
    ylab("") +
    ggtitle("Correlation Heatmap") +
    coord_fixed() +
    theme_minimal(base_size = 16) +
    theme(axis.text.x = element_text(angle = 90, 
                                     hjust = 1),
          legend.title = element_text(size = 12),
          legend.text = element_text(size = 10))
}
