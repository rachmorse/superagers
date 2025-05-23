if (!require("tidyr")) {
  install.packages("tidyr")
  require("tidyr")
}
if (!require("dplyr")) {
  install.packages("dplyr")
  require("dplyr")
}
if (!require("ggplot2")) {
  install.packages("ggplot2")
  require("ggplot2")
}
if (!require("lme4")) {
  install.packages("lme4")
  require("lme4")
}
if (!require("lmerTest")) {
  install.packages("lmerTest")
  require("lmerTest")
}
if (!require("emmeans")) {
  install.packages("emmeans")
  require("emmeans")
}
if (!require("ggpubr")) {
  install.packages("ggpubr")
  require("ggpubr")
}

# Read in data
data <- read.csv("~/Documents/2023:2024/Data/Exported data/clean_data_all.csv")

data <- data %>%
  mutate(id = as.numeric(gsub("sub-", "", id)))

# Create a cohort variable 
data$cohort <- ifelse(data$id > 5000, "bbhi", "bbhi_senior")

# Create a long df
long_data <- data %>%
  pivot_longer(
    cols = matches("^age\\_\\d$|^memory\\_\\d$|^sfc_all\\_\\d$|^func_all\\_\\d$|^struct_all\\_\\d$|^sfc_Hippocampus\\_\\d$|^struct_within_Default\\_\\d$|^func_within_DorsalAttention\\_\\d$|^func_between_Default\\_\\d$|^sfc_Default\\_\\d$|^sfc_Frontoparietal\\_\\d$|^func_within_Frontoparietal\\_\\d$|^func_within_Visual\\_\\d$"),
    names_to = c(".value", "timepoint"),
    names_pattern = "(.*)(\\d$)"
  )

long_data <- long_data %>%
  rename(
    age = "age_",
    memory = "memory_",
    sfc_all = "sfc_all_",
    func_all = "func_all_",
    struct_all = "struct_all_", 
    struct_within_Default = "struct_within_Default_",
    func_within_DorsalAttention = "func_within_DorsalAttention_",
    func_between_Default = "func_between_Default_",
    func_within_Frontoparietal = "func_within_Frontoparietal_",
    sfc_Frontoparietal = "sfc_Frontoparietal_",
    sfc_Hippocampus = "sfc_Hippocampus_",
    func_within_Visual = "func_within_Visual_",
    sfc_Default = "sfc_Default_"
  )

data <- data %>%
  mutate(superager_maintainer = case_when(
    maintainer == 1 & superager == 1 ~ "superager maintainer",
    # maintainer == 0 & superager == 1 ~ "superager decliner",
    # maintainer == 1 & superager == 0 ~ "non-superager maintainer",
    maintainer == 0 & superager == 0 ~ "non-superager decliner",
    TRUE ~ "unknown"
  ))

##########################################
####### Function to compare ANOVAs #######
##########################################

run_models <- function(df, independent_vars, dependent_vars, model_type = "aov") {
  # Suppress warnings
  oldw <- getOption("warn")
  options(warn = -1)
  
  # Validate model_type
  if (!model_type %in% c("aov", "lm")) {
    stop("model_type must be either 'aov' or 'lm'")
  }
  
  # Convert single grouping variable to list for consistency
  if (!is.list(independent_vars) && length(independent_vars) == 1) {
    independent_vars <- as.list(independent_vars)
  }
  
  # Initialize list to store results for each grouping variable
  all_results <- list()
  
  # Process each grouping variable
  for (independent_var in independent_vars) {
    
    results <- data.frame(
      dependent_var = character(),
      independent_var = character(),
      F_value = numeric(),
      p_value = numeric(),
      significant = logical(),
      stringsAsFactors = FALSE
    )
    
    for (dependent_var in dependent_vars) {
      # Build the formula as a string
      formula_str <- paste(dependent_var, "~", independent_var, "+ age_1 + YoE + sex")
      
      # Skip variables that don't exist in the dataframe
      if (!dependent_var %in% names(df)) {
        next  # Silent skip
      }
      
      # Attempt to fit the model based on model_type
      tryCatch({
        if (model_type == "aov") {
          fit <- aov(as.formula(formula_str), data = df)
          sumry <- summary(fit)[[1]]
          
          # Find the row for the independent variable in ANOVA table
          row_idx <- grep(paste0("^\\s*", independent_var), rownames(sumry))
          
          if (length(row_idx) == 1) {
            f_val <- sumry[row_idx, "F value"]
            p_val <- sumry[row_idx, "Pr(>F)"]
          } else {
            next  # Skip if can't find the row
          }
        } else if (model_type == "lm") {
          fit <- lm(as.formula(formula_str), data = df)
          sumry <- summary(fit)
          
          # Find the coefficient for the independent variable
          coef_idx <- grep(paste0("^", independent_var), names(coef(fit)))
          
          if (length(coef_idx) == 1) {
            # For lm, we extract the t-value and convert to F (t^2 = F with 1 df)
            t_val <- sumry$coefficients[coef_idx, "t value"]
            f_val <- t_val^2
            p_val <- sumry$coefficients[coef_idx, "Pr(>|t|)"]
          } else {
            next  # Skip if can't find the coefficient
          }
        }
        
        is_sig <- !is.na(p_val) && p_val < 0.05
        
        # Add to results dataframe
        results <- rbind(results, data.frame(
          dependent_var = dependent_var,
          independent_var = as.character(independent_var),
          F_value = f_val,
          p_value = p_val,
          significant = is_sig,
          stringsAsFactors = FALSE
        ))
        
        # Print significant results immediately
        if (is_sig) {
          if (model_type == "aov") {
            cat("Dependent variable:", dependent_var, "- F(", sumry[row_idx, "Df"], ",", 
                sumry["Residuals", "Df"], ") =", round(f_val, 2), 
                ", p =", format(p_val, digits = 3), "\n")
          } else {
            cat("Dependent variable:", dependent_var, "- t =", 
                round(sqrt(f_val), 2), " (F =", round(f_val, 2), 
                "), p =", format(p_val, digits = 3), "\n")
          }
        }
      }, error = function(e) {
        # Silent error handling
      })
    }
    
    # Sort results by p-value for this independent variable
    results <- results[order(results$p_value), ]
    
    # Add FDR correction for this independent variable
    if (nrow(results) > 0) {
      results$fdr_p_value <- p.adjust(results$p_value, method = "fdr")
      results$fdr_significant <- results$fdr_p_value < 0.05
    } else {
      results$fdr_p_value <- numeric(0)
      results$fdr_significant <- logical(0)
    }
    
    # Store in the list
    all_results[[as.character(independent_var)]] <- results
  } # End of for loop for independent_var
  
  # Restore warning level
  options(warn = oldw)
  
  # Combine all results
  combined_results <- do.call(rbind, all_results)
  rownames(combined_results) <- NULL
  
  # Sort the combined results by p-value
  combined_results <- combined_results[order(combined_results$p_value), ]
  
  # Apply FDR correction to the combined results
  if (nrow(combined_results) > 0) {
    combined_results$fdr_p_value <- p.adjust(combined_results$p_value, method = "fdr")
    combined_results$fdr_significant <- combined_results$fdr_p_value < 0.05
  }
  
  # Return and print summary of all results
  cat("\n\n==== SUMMARY OF ALL RESULTS ====\n")
  return(list(
    # individual_results = all_results,
    combined_results = combined_results
  ))
}

#####################################
########## Define variables #########
#####################################
vars_long_sfc <- c(
  "sfc_all_slopes",
  "sfc_Hippocampus_slopes",
  "sfc_Subcortical_slopes",
  "sfc_Default_slopes",
  "sfc_Frontoparietal_slopes",
  "sfc_VentralAttention_slopes",
  "sfc_DorsalAttention_slopes"
)

vars_cs_sfc_1 <- c(
  "sfc_Hippocampus_1",
  "sfc_Subcortical_1",
  "sfc_Default_1",
  "sfc_Frontoparietal_1",
  "sfc_VentralAttention_1",
  "sfc_DorsalAttention_1",
  "sfc_all_1"
)

vars_cs_sfc_2 <- c(
  "sfc_Hippocampus_2",
  "sfc_Subcortical_2",
  "sfc_Default_2",
  "sfc_Frontoparietal_2",
  "sfc_VentralAttention_2",
  "sfc_DorsalAttention_2",
  "sfc_all_2"
)

vars_long_struct <- c(
  "struct_all_slopes",
  "struct_Hippocampus_slopes",
  "struct_within_Subcortical_slopes",
  "struct_within_Default_slopes",
  "struct_within_Frontoparietal_slopes",
  "struct_within_VentralAttention_slopes",
  "struct_within_DorsalAttention_slopes",
  "struct_between_Subcortical_slopes",
  "struct_between_Default_slopes",
  "struct_between_Frontoparietal_slopes",
  "struct_between_VentralAttention_slopes",
  "struct_between_DorsalAttention_slopes",
  "struct_all_Subcortical_slopes",
  "struct_all_Default_slopes",
  "struct_all_Frontoparietal_slopes",
  "struct_all_VentralAttention_slopes",
  "struct_all_DorsalAttention_slopes"
)

vars_cs_struct_1 <- c(
  "struct_all_1",
  "struct_Hippocampus_1",
  "struct_within_Subcortical_1",
  "struct_within_Default_1",
  "struct_within_Frontoparietal_1",
  "struct_within_VentralAttention_1",
  "struct_within_DorsalAttention_1",
  "struct_between_Subcortical_1",
  "struct_between_Default_1",
  "struct_between_Frontoparietal_1",
  "struct_between_VentralAttention_1",
  "struct_between_DorsalAttention_1",
  "struct_all_Subcortical_1",
  "struct_all_Default_1",
  "struct_all_Frontoparietal_1",
  "struct_all_VentralAttention_1",
  "struct_all_DorsalAttention_1"
)

vars_cs_struct_2 <- c(
  "struct_all_2",
  "struct_Hippocampus_2",
  "struct_within_Subcortical_2",
  "struct_within_Default_2",
  "struct_within_Frontoparietal_2",
  "struct_within_VentralAttention_2",
  "struct_within_DorsalAttention_2",
  "struct_between_Subcortical_2",
  "struct_between_Default_2",
  "struct_between_Frontoparietal_2",
  "struct_between_VentralAttention_2",
  "struct_between_DorsalAttention_2",
  "struct_all_Subcortical_2",
  "struct_all_Default_2",
  "struct_all_Frontoparietal_2",
  "struct_all_VentralAttention_2",
  "struct_all_DorsalAttention_2"
)

vars_long_func <- c(
  "func_all_slopes",
  "func_Hippocampus_slopes",
  "func_within_Subcortical_slopes",
  "func_within_Default_slopes",
  "func_within_Frontoparietal_slopes",
  "func_within_VentralAttention_slopes",
  "func_within_DorsalAttention_slopes",
  "func_between_Subcortical_slopes",
  "func_between_Default_slopes",
  "func_between_Frontoparietal_slopes",
  "func_between_VentralAttention_slopes",
  "func_between_DorsalAttention_slopes",
  "func_all_Subcortical_slopes",
  "func_all_Default_slopes",
  "func_all_Frontoparietal_slopes",
  "func_all_VentralAttention_slopes",
  "func_all_DorsalAttention_slopes"
)

vars_cs_func_1 <- c(
  "func_all_1",
  "func_Hippocampus_1",
  "func_within_Subcortical_1",
  "func_within_Default_1",
  "func_within_Frontoparietal_1",
  "func_within_VentralAttention_1",
  "func_within_DorsalAttention_1",
  "func_between_Subcortical_1",
  "func_between_Default_1",
  "func_between_Frontoparietal_1",
  "func_between_VentralAttention_1",
  "func_between_DorsalAttention_1",
  "func_all_Subcortical_1",
  "func_all_Default_1",
  "func_all_Frontoparietal_1",
  "func_all_VentralAttention_1",
  "func_all_DorsalAttention_1"
)

vars_cs_func_2 <- c(
  "func_all_2",
  "func_Hippocampus_2",
  "func_within_Subcortical_2",
  "func_within_Default_2",
  "func_within_Frontoparietal_2",
  "func_within_VentralAttention_2",
  "func_within_DorsalAttention_2",
  "func_between_Subcortical_2",
  "func_between_Default_2",
  "func_between_Frontoparietal_2",
  "func_between_VentralAttention_2",
  "func_between_DorsalAttention_2",
  "func_all_Subcortical_2",
  "func_all_Default_2",
  "func_all_Frontoparietal_2",
  "func_all_VentralAttention_2",
  "func_all_DorsalAttention_2"
)

#####################################
########## SFC, struct, func, memory ###########
#####################################

# SFC and memory
run_models(data, vars_long_sfc, "memory_slopes", "lm")
run_models(data, vars_cs_sfc_1, "memory_1", "lm")
run_models(data, vars_cs_sfc_2, "memory_2", "lm")

# Structure and memory
run_models(data, vars_long_struct, "memory_slopes", "lm")
run_models(data, vars_cs_struct_1, "memory_1", "lm")
run_models(data, vars_cs_struct_2, "memory_2", "lm")

# Functional and memory
run_models(data, vars_long_func, "memory_slopes", "lm")
run_models(data, vars_cs_func_1, "memory_1", "lm")
run_models(data, vars_cs_func_2, "memory_2", "lm")


ggplot(data, aes(x = sfc_Hippocampus_slopes, y = memory_slopes, color = factor(superager))) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, aes(group = factor(superager))) +
  labs(
    x = "SFC Hippocampus Slopes",
    y = "Memory Slopes",
    color = "Superager Status"
  ) +
  scale_color_manual(values = c("0" = "#E69F00", "1" = "#0072B2"), 
                     labels = c("0" = "Non-superager", "1" = "Superager")) +
  theme_minimal()

ggplot(data, aes(x = factor(superager), y = memory_slopes, fill = factor(superager))) +
  geom_violin(trim = FALSE, alpha = 0.7) +
  geom_boxplot(width = 0.2, position = position_dodge(0.9), alpha = 0.6) +
  geom_jitter(width = 0.1, alpha = 0.5) +  # Add individual data points
  stat_summary(fun = mean, geom = "point", shape = 18, size = 4, color = "black") +
  labs(
    x = "",
    y = "Memory Slopes"
  ) +
  theme_minimal() +
  theme(legend.position = "none") +
  stat_compare_means(method = "anova", 
                     label.y = max(data$memory_slopes, na.rm = TRUE) * 1.1)

# 1) Fit the model
model_memory_lme <- lmer(memory ~ age * factor(superager) + (1 | id) + cohort  + sex + YoE, data = long_data)
summary(model_memory_lme)

# Center the variables by subtracting the mean
long_data_bbhi$memory_centered = scale(long_data_bbhi$memory)
long_data_bbhi$sfc_Hippocampus_centered = scale(long_data_bbhi$sfc_Hippocampus)

model_memory_lme <- lmer(memory_centered ~ sfc_Hippocampus_centered * factor(maintainer) + (1 | id)  + sex + YoE, data = long_data_bbhi)
summary(model_memory_lme)

summary(lmer(scale(struct_all) ~  scale(age) * factor(superager) + (1 | id)  + sex + YoE, data = long_data_bbhi))
summary(lmer(scale(memory) ~  scale(struct_all) * scale(superager) * scale(age) + (1 | id) + sex + YoE, data = long_data_bbhi))

summary(lmer(scale(memory) ~  scale(func_all) * scale(maintainer) * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all) * scale(maintainer) * scale(age) + (1 | id) + sex + YoE, data = long_data_bbhi))
summary(lmer(scale(memory) ~  scale(sfc_all) * scale(maintainer) * scale(age) + (1 | id) + sex + YoE, data = long_data_bbhi))
summary(lmer(scale(func_all) ~  scale(maintainer) * scale(age) + (1 | id) + sex + YoE, data = long_data))


summary(lmer(scale(memory) ~  scale(func_all) * scale(superager) * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all) * scale(superager) * scale(age) + (1 | id) + sex + YoE, data = long_data_bbhi))
summary(lmer(scale(memory) ~  scale(sfc_all) * scale(superager) * scale(age) + (1 | id) + sex + YoE, data = long_data_bbhi))
summary(lmer(scale(func_all) ~  scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all) * scale(superager) + (1 | id) + age + sex + YoE + cohort, data = long_data))


long_data$age_centered = scale(long_data$age)
long_data$func_all_centered = scale(long_data$func_all)
long_data$memory_centered = scale(long_data$memory)

superager_long <- long_data %>% 
  filter(superager == 1)
nonsuperager_long <- long_data %>% 
  filter(superager == 0)

model_memory_func <- lmer((memory) ~  (func_all) + (1 | id) +  age + sex + YoE, data = nonsuperager_long)
summary(model_memory_func)
model_memory_func <- lmer((memory) ~  (func_all) + (1 | id) +  age + sex + YoE, data = superager_long)
summary(model_memory_func)
model_memory_func <- lmer(scale(memory) ~  scale(func_all) * scale(superager) + (1 | id) + age + sex + YoE + cohort, data = long_data)
summary(model_memory_func)

model_memory_lme <- lmer((func_all_centered) ~ (age_centered) * factor(superager) + (1 | id)  + sex + YoE, data = long_data)
summary(model_memory_lme)

# 2) Create a grid of ages for which we want predictions
age_seq <- seq(
  from = min(long_data$age_centered, na.rm = TRUE),
  to   = max(long_data$age_centered, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each superager_factor × age
em_2group <- emmeans(
  model_memory_lme, 
  specs = ~ factor(superager) * age_centered,
  at    = list(age_centered = age_seq),
  reff  = 0 # set reff = 0 to ignore random effects (similar to re.form = NA).
)

# 4) Convert the emmeans results to a data frame and rename columns
predictions <- summary(em_2group) %>% 
  as.data.frame() %>% 
  rename(
    predicted   = emmean,
    lower_bound = lower.CL,
    upper_bound = upper.CL
  ) %>% 
  mutate(age_centered = as.numeric(age_centered))  # Convert from factor if needed

# Define colors for each group
palette_1 <- c("1" = "#A35C7A", "0" = "#FFD65A")

# Create the two-group plot
ggplot() +
  geom_line(
    data = long_data, 
    aes(x = age_centered, y = func_all_centered, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data, 
    aes(x = age_centered, y = func_all_centered), 
    color = "gray", size = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = age_centered, ymin = lower_bound, ymax = upper_bound, 
        fill = factor(superager)), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = age_centered, y = predicted, color = factor(superager)), 
    size = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "age", y = "func_all", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))

# 2) Create a grid of ages for which we want predictions
func_all_centered_seq <- seq(
  from = min(long_data$func_all_centered, na.rm = TRUE),
  to   = max(long_data$func_all_centered, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for age only
em_overall <- emmeans(
  model_memory_func, 
  specs = ~ func_all_centered,
  at    = list(func_all_centered = func_all_centered_seq),
  reff  = 0 # set reff = 0 to ignore random effects
)

# 4) Convert the emmeans results to a data frame and rename columns
predictions <- summary(em_overall) %>% 
  as.data.frame() %>% 
  rename(
    predicted   = emmean,
    lower_bound = lower.CL,
    upper_bound = upper.CL
  ) %>% 
  mutate(func_all_centered = as.numeric(func_all_centered))  
line_color <- "#3366CC" 

# Create the single trend line plot
ggplot() +
  geom_line(
    data = long_data, 
    aes(x = func_all_centered, y = memory_centered, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data, 
    aes(x = func_all_centered, y = memory_centered), 
    color = "gray", size = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = func_all_centered, ymin = lower_bound, ymax = upper_bound), 
    fill = line_color, alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = func_all_centered, y = predicted), 
    color = line_color, size = 1.5
  ) +  # Plot predicted line
  labs(x = "func_all_centered", y = "memory_centered") +
  theme_minimal() +
  theme(legend.position = "none") 


ggplot(data, aes(x = func_all_DorsalAttention_slopes, y = memory_slopes)) +
  geom_point(color = "darkblue", size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(
    x = "func_all_DorsalAttention_slopes",
    y = "Memory Slopes"
  ) +
  theme_minimal() 

ggplot(data, aes(x = func_all_VentralAttention_slopes, y = memory_slopes)) +
  geom_point(color = "darkblue", size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(
    x = "func_all_VentralAttention_slopes",
    y = "Memory Slopes"
  ) +
  theme_minimal() 

# Because of this, does superager / maintainer status change this relationship 
summary(lm(scale(memory_slopes) ~ scale(superager) * scale(func_all_VentralAttention_slopes) + scale(age_1) + scale(YoE) + sex, data))

####################################
####### ANOVAs FC #######
####################################

run_models(data, "superager", vars_long_func, "aov")
run_models(data, "superager", vars_cs_func_1, "aov")
run_models(data, "superager", vars_cs_func_2, "aov")

run_models(data, "maintainer", vars_long_func, "aov")
run_models(data, "maintainer", vars_cs_func_1, "aov")
run_models(data, "maintainer", vars_cs_func_2, "aov")

run_models(data, "superager_maintainer", vars_long_func, "aov")
run_models(data, "superager_maintainer", vars_cs_func_1, "aov")
run_models(data, "superager_maintainer", vars_cs_func_2, "aov")

# Plot 
data$maintainer_group <- factor(data$maintainer, 
                                levels = c(0, 1), 
                                labels = c("Decline", "Maintainer"))

ggplot(data, aes(x = maintainer_group, y = func_within_VentralAttention_slopes, fill = maintainer_group)) +
  geom_violin(trim = FALSE, alpha = 0.7) +
  geom_boxplot(width = 0.2, position = position_dodge(0.9), alpha = 0.6) +
  geom_jitter(width = 0.1, alpha = 0.5) +  # Add individual data points
  stat_summary(fun = mean, geom = "point", shape = 18, size = 4, color = "black") +
  labs(
    x = "",
    y = "FC VentralAttention Slopes"
  ) +
  theme_minimal() +
  theme(legend.position = "none") +
  stat_compare_means(method = "anova", 
                     label.y = max(data$func_within_VentralAttention_slopes, na.rm = TRUE) * 1.1)

####################################
####### ANOVAs SC #######
####################################

run_models(data, "superager", vars_long_struct, "aov")
run_models(data, "superager", vars_cs_struct_1, "aov")
run_models(data, "superager", vars_cs_struct_2, "aov")

run_models(data, "maintainer", vars_long_struct, "aov")
run_models(data, "maintainer", vars_cs_struct_1, "aov")
run_models(data, "maintainer", vars_cs_struct_2, "aov")

run_models(data, "superager_maintainer", vars_long_struct, "aov")
run_models(data, "superager_maintainer", vars_cs_struct_1, "aov")
run_models(data, "superager_maintainer", vars_cs_struct_2, "aov")

####################################
####### ANOVAs SFC #######
####################################

run_models(data, "superager", vars_long_sfc, "aov")
run_models(data, "superager", vars_cs_sfc_1, "aov")
run_models(data, "superager", vars_cs_sfc_2, "aov")

run_models(data, "maintainer", vars_long_sfc, "aov")
run_models(data, "maintainer", vars_cs_sfc_1, "aov")
run_models(data, "maintainer", vars_cs_sfc_2, "aov")

run_models(data, "superager_maintainer", vars_long_sfc, "aov")
run_models(data, "superager_maintainer", vars_cs_sfc_1, "aov")
run_models(data, "superager_maintainer", vars_cs_sfc_2, "aov")

# Plot 
ggplot(data, aes(x = maintainer_group, y = sfc_Default_slopes, fill = maintainer_group)) +
  geom_violin(trim = FALSE, alpha = 0.7) +
  geom_boxplot(width = 0.2, position = position_dodge(0.9), alpha = 0.6) +
  geom_jitter(width = 0.1, alpha = 0.5) +  # Add individual data points
  stat_summary(fun = mean, geom = "point", shape = 18, size = 4, color = "black") +
  labs(
    x = "",
    y = "SFC Slopes"
  ) +
  theme_minimal() +
  theme(legend.position = "none") +
  stat_compare_means(method = "anova", 
                     label.y = max(data$func_within_VentralAttention_slopes, na.rm = TRUE) * 1.1)

##################################
########## SFC and age ###########
##################################

# Filter to those with two tps
long_data_bbhi <- long_data %>% 
  filter(cohort == "bbhi")

# Run lm 
lm_sfc_age <- lmer(scale(sfc_all) ~ scale(age) + (1 | id) + sex + scale(YoE), data = long_data_bbhi)
summary(lm_sfc_age)
lm_sfc_age <- lmer((sfc_all) ~ (age) + (1 | id) + sex + (YoE), data = long_data_bbhi)
summary(lm_sfc_age)

# 2) Create a grid of ages for which we want predictions
age_seq <- seq(
  from = min(long_data_bbhi$age, na.rm = TRUE),
  to   = max(long_data_bbhi$age, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for age only
em_overall <- emmeans(
  lm_sfc_age, 
  specs = ~ age,
  at    = list(age = age_seq),
  reff  = 0 # set reff = 0 to ignore random effects
)

# 4) Convert the emmeans results to a data frame and rename columns
predictions <- summary(em_overall) %>% 
  as.data.frame() %>% 
  rename(
    predicted   = emmean,
    lower_bound = lower.CL,
    upper_bound = upper.CL
  ) %>% 
  mutate(age = as.numeric(age))  
line_color <- "#3366CC" 

# Create the single trend line plot
ggplot() +
  geom_line(
    data = long_data_bbhi, 
    aes(x = age, y = sfc_all, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data_bbhi, 
    aes(x = age, y = sfc_all), 
    color = "gray", size = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = age, ymin = lower_bound, ymax = upper_bound), 
    fill = line_color, alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = age, y = predicted), 
    color = line_color, size = 1.5
  ) +  # Plot predicted line
  labs(x = "Age", y = "SFC") +
  theme_minimal() +
  theme(legend.position = "none") 

# Filter to those with two tps
long_data_bbhi <- long_data %>% 
  filter(cohort == "bbhi")


#################################
######### Interactions ##########
#################################

data$superager_group <- factor(data$superager, 
                               levels = c(0, 1), 
                               labels = c("Non-Superager", "Superager"))
# Plot
ggplot(
  data,
  aes(
    x = sfc_Frontoparietal_slopes,
    y = memory_slopes,
    color = factor(superager)
  )
) +
  geom_point(alpha = 0.5) +
  geom_smooth(
    method = "lm",
    aes(group = factor(superager)),
    se = TRUE
  ) +
  labs(
    y = "Memory Slopes",
    x = "SFC Frontoparietal Slopes",
    color = "Superager status"
  ) +
  theme_minimal() +
  theme(
    legend.position = "right",
    plot.background = element_rect(fill = "white", color = "white"),
    plot.margin = margin(t = 0.35, r = 0.35, b = 0.35, l = 0.35, unit = "cm"),
    axis.text.x = element_text(margin = margin(t = -5)),
    plot.title = element_text(hjust = 0.5)
  )

vars_long_sfc <- c(
  "sfc_all_slopes",
  "sfc_Hippocampus_slopes",
  "sfc_Subcortical_slopes",
  "sfc_Default_slopes",
  "sfc_Frontoparietal_slopes",
  "sfc_VentralAttention_slopes",
  "sfc_DorsalAttention_slopes"
)

summary(lm(scale(memory_slopes) ~ scale(superager) * scale(sfc_all_slopes) + scale(age_1) + scale(YoE) + sex, data))
summary(lm(scale(memory_slopes) ~ scale(superager) * scale(sfc_Hippocampus_slopes) + scale(age_1) + scale(YoE) + sex, data))
summary(lm(scale(memory_slopes) ~ scale(superager) * scale(sfc_Subcortical_slopes) + scale(age_1) + scale(YoE) + sex, data))
summary(lm(scale(memory_slopes) ~ scale(superager) * scale(sfc_Default_slopes) + scale(age_1) + scale(YoE) + sex, data))
summary(lm(scale(memory_slopes) ~ scale(superager) * scale(sfc_Frontoparietal_slopes) + scale(age_1) + scale(YoE) + sex, data))
summary(lm(scale(memory_slopes) ~ scale(superager) * scale(sfc_VentralAttention_slopes) + scale(age_1) + scale(YoE) + sex, data))
summary(lm(scale(memory_slopes) ~ scale(superager) * scale(sfc_DorsalAttention_slopes) + scale(age_1) + scale(YoE) + sex, data))

##########################
# SFC interactions 

# Create a list to store the models
models <- list(
  all = lm(scale(memory_slopes) ~ scale(maintainer) * scale(sfc_all_slopes) + scale(age_1) + scale(YoE) + sex, data),
  default = lm(scale(memory_slopes) ~ scale(maintainer) * scale(sfc_Default_slopes) + scale(age_1) + scale(YoE) + sex, data),
  frontoparietal = lm(scale(memory_slopes) ~ scale(maintainer) * scale(sfc_Frontoparietal_slopes) + scale(age_1) + scale(YoE) + sex, data),
  ventral = lm(scale(memory_slopes) ~ scale(maintainer) * scale(sfc_VentralAttention_slopes) + scale(age_1) + scale(YoE) + sex, data),
  dorsal = lm(scale(memory_slopes) ~ scale(maintainer) * scale(sfc_DorsalAttention_slopes) + scale(age_1) + scale(YoE) + sex, data)
)

# Extract interaction p-values
interaction_terms <- c("scale(superager):scale(sfc_all_slopes)",
                       "scale(superager):scale(sfc_Default_slopes)",
                       "scale(superager):scale(sfc_Frontoparietal_slopes)",
                       "scale(superager):scale(sfc_VentralAttention_slopes)",
                       "scale(superager):scale(sfc_DorsalAttention_slopes)")

# Create a data frame to store results
results <- data.frame(
  network = c("All Networks", "Default Mode", "Frontoparietal", "Ventral Attention", "Dorsal Attention"),
  p_value = numeric(5),
  significant = logical(5),
  stringsAsFactors = FALSE
)

# Extract p-values for interaction terms
for (i in 1:length(models)) {
  model_summary <- summary(models[[i]])
  coef_table <- model_summary$coefficients
  
  # Find the interaction term (may have different names depending on your data)
  interaction_row <- which(rownames(coef_table) == interaction_terms[i])
  
  if (length(interaction_row) > 0) {
    results$p_value[i] <- coef_table[interaction_row, "Pr(>|t|)"]
    results$significant[i] <- results$p_value[i] < 0.05
  }
}

# Apply FDR correction
results$fdr_p_value <- p.adjust(results$p_value, method = "fdr")
results$fdr_significant <- results$fdr_p_value < 0.05

# Display the results with original and FDR-corrected p-values
print(results[order(results$p_value), c("network", "p_value", "significant", "fdr_p_value", "fdr_significant")])

##########################
# structure interactions 

# Create a list to store the models
models <- list(
  all = lm(scale(memory_slopes) ~ scale(superager) * scale(struct_all_slopes) + scale(age_1) + scale(YoE) + sex, data),
  default = lm(scale(memory_slopes) ~ scale(superager) * scale(struct_within_Default_slopes) + scale(age_1) + scale(YoE) + sex, data),
  frontoparietal = lm(scale(memory_slopes) ~ scale(superager) * scale(struct_within_Frontoparietal_slopes) + scale(age_1) + scale(YoE) + sex, data),
  ventral = lm(scale(memory_slopes) ~ scale(superager) * scale(struct_within_VentralAttention_slopes) + scale(age_1) + scale(YoE) + sex, data),
  dorsal = lm(scale(memory_slopes) ~ scale(superager) * scale(struct_within_DorsalAttention_slopes) + scale(age_1) + scale(YoE) + sex, data)
)

# Extract interaction p-values
interaction_terms <- c("scale(superager):scale(struct_all_slopes)",
                       "scale(superager):scale(struct_within_Default_slopes)",
                       "scale(superager):scale(struct_within_Frontoparietal_slopes)",
                       "scale(superager):scale(struct_within_VentralAttention_slopes)",
                       "scale(superager):scale(struct_within_DorsalAttention_slopes)")

# Create a data frame to store results
results <- data.frame(
  network = c("All Networks", "Default Mode", "Frontoparietal", "Ventral Attention", "Dorsal Attention"),
  p_value = numeric(5),
  significant = logical(5),
  stringsAsFactors = FALSE
)

# Extract p-values for interaction terms
for (i in 1:length(models)) {
  model_summary <- summary(models[[i]])
  coef_table <- model_summary$coefficients
  
  # Find the interaction term (may have different names depending on your data)
  interaction_row <- which(rownames(coef_table) == interaction_terms[i])
  
  if (length(interaction_row) > 0) {
    results$p_value[i] <- coef_table[interaction_row, "Pr(>|t|)"]
    results$significant[i] <- results$p_value[i] < 0.05
  }
}

# Apply FDR correction
results$fdr_p_value <- p.adjust(results$p_value, method = "fdr")
results$fdr_significant <- results$fdr_p_value < 0.05

# Display the results with original and FDR-corrected p-values
print(results[order(results$p_value), c("network", "p_value", "significant", "fdr_p_value", "fdr_significant")])

##########################
# functional interactions 

# Create a list to store the models
models <- list(
  all = lm(scale(memory_slopes) ~ scale(superager) * scale(func_all_slopes) + scale(age_1) + scale(YoE) + sex, data),
  default = lm(scale(memory_slopes) ~ scale(superager) * scale(func_within_Default_slopes) + scale(age_1) + scale(YoE) + sex, data),
  frontoparietal = lm(scale(memory_slopes) ~ scale(superager) * scale(func_within_Frontoparietal_slopes) + scale(age_1) + scale(YoE) + sex, data),
  ventral = lm(scale(memory_slopes) ~ scale(superager) * scale(func_within_VentralAttention_slopes) + scale(age_1) + scale(YoE) + sex, data),
  dorsal = lm(scale(memory_slopes) ~ scale(superager) * scale(func_within_DorsalAttention_slopes) + scale(age_1) + scale(YoE) + sex, data)
)

# Extract interaction p-values
interaction_terms <- c("scale(superager):scale(func_all_slopes)",
                       "scale(superager):scale(func_within_Default_slopes)",
                       "scale(superager):scale(func_within_Frontoparietal_slopes)",
                       "scale(superager):scale(func_within_VentralAttention_slopes)",
                       "scale(superager):scale(func_within_DorsalAttention_slopes)")

# Create a data frame to store results
results <- data.frame(
  network = c("All Networks", "Default Mode", "Frontoparietal", "Ventral Attention", "Dorsal Attention"),
  p_value = numeric(5),
  significant = logical(5),
  stringsAsFactors = FALSE
)

# Extract p-values for interaction terms
for (i in 1:length(models)) {
  model_summary <- summary(models[[i]])
  coef_table <- model_summary$coefficients
  
  # Find the interaction term (may have different names depending on your data)
  interaction_row <- which(rownames(coef_table) == interaction_terms[i])
  
  if (length(interaction_row) > 0) {
    results$p_value[i] <- coef_table[interaction_row, "Pr(>|t|)"]
    results$significant[i] <- results$p_value[i] < 0.05
  }
}

# Apply FDR correction
results$fdr_p_value <- p.adjust(results$p_value, method = "fdr")
results$fdr_significant <- results$fdr_p_value < 0.05

# Display the results with original and FDR-corrected p-values
print(results[order(results$p_value), c("network", "p_value", "significant", "fdr_p_value", "fdr_significant")])
