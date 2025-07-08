# Exercise - Ames data EDA

rm(list = ls())
graphics.off()

# Install packages
#install.packages("janitor")

# Load packages
library(tidyverse)
library(modeldata)
library(janitor)

# Paths
path_figures <- "./fig/"


# Data

## Ames housing prices
df <- modeldata::ames %>% 
  janitor::clean_names()


# EDA

## data set size
df %>% ncol() # number of variables
df %>% nrow() # number of units (diamonds) - measurements

## columns list (types included)
df.cols <- map(.x = df, 
               .f = class) %>% 
  unlist() %>% 
  tibble(col = names(.),
         type = .) %>% 
  mutate(position = row_number(),
         role = if_else(col == "sale_price", 
                        "target", 
                        "feature")) %>% 
  arrange(desc(role), position)



## Initial EDA

### column types count - figure
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

ggsave(filename = paste0(path_figures, "01_col_types.png"), plot = last_plot(), device = "png", 
       units = "cm", width = 22, height = 15, dpi = 600)


### check missing values
map(.x = df,
    .f = ~sum(is.na(.))) %>% 
  unlist() %>% 
  tibble(col = names(.),
         nr_missing = .) %>% 
  ggplot(aes(y = col,
             x = nr_missing)) +
  geom_point(size = 2,
             color = "brown1") +
  ggtitle("Number of missing rows by column") +
  xlab("Number of missing rows") +
  ylab("Column") +
  theme_minimal()

ggsave(filename = paste0(path_figures, "01_miss_val_count.png"), plot = last_plot(), device = "png", 
       units = "cm", width = 15, height = 22, dpi = 600)



## Distribution of target variable

### value "as is"
df %>% 
  ggplot(aes(x = sale_price)) +
  geom_density(fill = "gray80",
               color = "black",
               alpha = 0.6) +
  xlab("Sale price in USD") +
  ylab("Density") +
  ggtitle("Property prices - target variable") + 
  theme_minimal(base_size = 16)

ggsave(filename = paste0(path_figures, "02_tar_var_dist.png"), plot = last_plot(), device = "png", 
       units = "cm", width = 22, height = 15, dpi = 600)

### value "log10 transformed"
df %>% 
  ggplot(aes(x = log10(sale_price))) +
  geom_density(fill = "gray80",
               color = "black",
               alpha = 0.6) +
  xlab("Sale price in USD - log10") +
  ylab("Density") +
  ggtitle("Property prices - target variable") + 
  theme_minimal(base_size = 16)

ggsave(filename = paste0(path_figures, "02_tar_var_log10_dist.png"), plot = last_plot(), device = "png", 
       units = "cm", width = 22, height = 15, dpi = 600)



## Factor features - unique levels & distributions

### Number of categories / levels - categorical (factors) features 
df %>% 
  select(where(is.factor)) %>% 
  map_int(., nlevels) %>% 
  enframe(name = "col", 
          value = "n") %>% 
  arrange(n) %>% 
  ggplot(aes(x = n,
             y = fct_inorder(col))) +
  geom_col(color = "black",
           fill = "gray80") +
  scale_x_continuous(breaks = c(seq(0,5), seq(10,30,5))) +
  ggtitle("Factor features - unique levels count") +
  xlab("Number of unique levels (categories)") +
  ylab("Feature") +
  theme_minimal(base_size = 16)

ggsave(filename = paste0(path_figures, "03_feat_fact_lev_count.png"), plot = last_plot(), device = "png", 
       units = "cm", width = 18, height = 22, dpi = 600)


### Most frequent level percentage 
df %>% 
  select(where(is.factor)) %>% 
  pivot_longer(cols = everything(), 
               names_to = "feature", 
               values_to = "level") %>%
  group_by(feature, level) %>%
  summarise(count = n(), 
            .groups = "drop") %>%
  group_by(feature) %>%
  mutate(per = count / sum(count) * 100) %>%
  slice_max(order_by = count, n = 1, 
            with_ties = FALSE) %>%
  ungroup() %>% 
  arrange(per) %>% 
  ggplot(aes(x = per,
             y = fct_inorder(feature),
             label = level)) +
  geom_col(color = "black",
           fill = "gray80") +
  geom_text(size = 4,
            aes(x = per - 5)) +
  scale_x_continuous(breaks = seq(0,100,10)) +
  labs(title = "Factor feature - top factor level percentage",
       subtitle = "Text on each bar shows name of top level in each factor feature") +
  xlab("Top factor level percentage of rows") +
  ylab("Feature") +
  theme_minimal(base_size = 16)

ggsave(filename = paste0(path_figures, "03_feat_fact_top_lev_per.png"), plot = last_plot(), device = "png", 
       units = "cm", width = 22, height = 22, dpi = 600)


### Factor features distribution 

fact_vars <- df.cols %>% filter(type == "factor") %>% pull(col) # select only factor columns

map(fact_vars, function(var_name) {
  p <- ggplot(df, 
              aes_string(x = var_name)) +
    geom_bar(color = "black",
             fill = "gray80") +
    labs(title = paste0("Distribution of values - ", var_name), 
         x = "", 
         y = "Frequency") +
    theme_minimal(base_size = 14) +
    theme(axis.text.x = element_text(angle = 90,
                                     size = 8, 
                                     hjust = 1))
  
  ggsave(filename = paste0(path_figures, "03_feat_fact_distr_", var_name, ".png"),
         plot = p, device = "png", units = "cm", 
         width = 22, height = 15, dpi = 600)
})



## Numeric features distributions

### Distribution density plot
num_vars <- df.cols %>% filter(type == "numeric" & role != "target") %>% pull(col)

map(num_vars, function(var_name) {
  p <- ggplot(df, 
              aes_string(x = var_name)) +
    geom_density(fill = "gray80", 
                 color = "black",
                 alpha = 0.7) +
    labs(title = paste("Feature distribution (density plot) - ", var_name), 
         x = var_name,
         y = "Value") +
    theme_minimal(base_size = 14)
  
  ggsave(filename = paste0(path_figures, "04_feat_num_distr_dens_plot_", var_name, ".png"),
         plot = p, device = "png", units = "cm", 
         width = 22, height = 15, dpi = 600)
})

### Distribution box plot

map(num_vars, function(var_name) {
  p <- ggplot(df, 
              aes_string(y = var_name)) +
    geom_boxplot() +
    labs(title = paste("Feature distribution (box plot) - ", var_name), 
         x = var_name,
         y = "Value") +
    theme_minimal(base_size = 14)
  
  ggsave(filename = paste0(path_figures, "04_feat_num_distr_box_plot_", var_name, ".png"),
         plot = p, device = "png", units = "cm", 
         width = 22, height = 15, dpi = 600)
})



## Integer features distributions

### Distribution histogram plot
int_vars <- df.cols %>% filter(type == "integer" & role != "target") %>% pull(col)

map(int_vars, function(var_name) {
  p <- ggplot(df, 
              aes_string(x = var_name)) +
    geom_histogram(fill = "gray80", 
                   color = "black",
                   alpha = 0.7, 
                   bins = 30) +
    labs(title = paste("Feature distribution (histogram) - ", var_name), 
         x = var_name,
         y = "Count") +
    theme_minimal(base_size = 14)
  
  ggsave(filename = paste0(path_figures, "05_feat_int_distr_hist_plot_", var_name, ".png"),
         plot = p, device = "png", units = "cm", 
         width = 22, height = 15, dpi = 600)
})



## Factor features VS target variable - box plots

fact_vars <- df.cols %>% filter(type == "factor") %>% pull(col) # select only factor columns

map(fact_vars, function(var_name) {
  p <- ggplot(df, 
              aes_string(x = var_name,
                         y = "sale_price")) +
    geom_boxplot() +
    labs(title = paste("Factor feature VS sale price box plot - ", var_name), 
         x = var_name,
         y = "Sale price in USD") +
    theme_minimal(base_size = 14) +
    theme(axis.text.x = element_text(angle = 90,
                                     size = 8, 
                                     hjust = 1))
  
  ggsave(filename = paste0(path_figures, "06_feat_fact_VS_tar_box_plot_", var_name, ".png"),
         plot = p, device = "png", units = "cm", 
         width = 22, height = 15, dpi = 600)
})



## All numeric features VS target variable - scatter plot

all_num_vars <- df.cols %>% filter(type != "factor" & role != "target") %>% pull(col)

map(all_num_vars, function(var_name) {
  p <- ggplot(df, 
              aes_string(x = var_name,
                         y = "sale_price")) +
    geom_jitter() +
    labs(title = paste("Numeric feature VS sale price scatter plot - ", var_name), 
         x = var_name,
         y = "Sale price in USD") +
    theme_minimal(base_size = 14)
  
  ggsave(filename = paste0(path_figures, "07_feat_num_int_VS_tar_scatter_", var_name, ".png"),
         plot = p, device = "png", units = "cm", 
         width = 22, height = 15, dpi = 600)
})



## Property longitude & latitude

## longitude & latitude VS neighborhood
df %>% 
  ggplot(aes(x = longitude,
             y = latitude,
             color = neighborhood)) +
  geom_point() +
  scale_color_viridis_d(option = "inferno") +
  xlab("Longitude (property)") +
  ylab("Latitude (property)") +
  labs(title = "Property's location VS neighborhood",
       color = "Neighborhood:") +
  theme_minimal(base_size = 14)

ggsave(filename = paste0(path_figures, "08_property_long_lat_neighborhood.png"), plot = last_plot(), device = "png", 
       units = "cm", width = 35, height = 25, dpi = 600)

## longitude & latitude VS property price
df %>% 
  ggplot(aes(x = longitude,
             y = latitude,
             color = sale_price)) +
  geom_point() +
  scale_color_viridis_c(option = "inferno") +
  xlab("Longitude (property)") +
  ylab("Latitude (property)") +
  labs(title = "Property's location VS sale price",
       color = "Sale price:") +
  theme_minimal(base_size = 14)

ggsave(filename = paste0(path_figures, "08_property_long_lat_price.png"), plot = last_plot(), device = "png", 
       units = "cm", width = 27, height = 25, dpi = 600)



## Correlation between numeric variables

### compute correlation matrix (Person coefficient)
cor_matrix <- df %>%
  select(one_of(df.cols %>% 
                  filter(type != "factor" & role != "target") %>% 
                  pull(col))) %>%
  cor(method = "pearson")

### convert to long format for ggplot
cor_long <- cor_matrix %>%
  as.data.frame() %>%
  rownames_to_column("var1") %>%
  pivot_longer(-var1, names_to = "var2", values_to = "correlation")

### plot the heatmap
ggplot(cor_long, 
       aes(x = var1, 
           y = var2, 
           fill = correlation)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(correlation, 2)), 
            size = 2.5) +
  scale_fill_gradient2(low = "blue", 
                       mid = "white", 
                       high = "red",
                       midpoint = 0, 
                       limit = c(-1, 1), 
                       name = "Pearson\nCorrelation") +
  xlab("") +
  ylab("") +
  ggtitle("Correlation Heatmap") +
  coord_fixed() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, 
                                   hjust = 1))

ggsave(filename = paste0(path_figures, "09_var_corr_heat_map.png"), plot = last_plot(), device = "png", 
       units = "cm", width = 30, height = 30, dpi = 600)
