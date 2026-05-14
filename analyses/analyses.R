suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(lme4)
  library(lmerTest)
  library(ggplot2)
  library(readxl)
  library(MuMIn)
})

#################
# Data cleaning #
#################

# Load source data
raw_data <- read.csv("~/Documents/2023:2024/Data/Exported data/superager.csv")
sfc_data_1 <- read.csv("~/Documents/2023:2024/Data/Exported data/all_sfc_data_ses-01.csv")
sfc_data_2 <- read.csv("~/Documents/2023:2024/Data/Exported data/all_sfc_data_ses-02.csv")
weighted <- read.csv("~/Documents/2023:2024/Data/Exported data/weighted_global_roi_averages.csv")
bbhi_senior <- readxl::read_excel("~/Documents/2023:2024/Data/BBHI-Senior/bbhi senior data.xlsx", .name_repair = "unique_quiet")
bbhi <- read.csv("~/Documents/2023:2024/Data/BBHI/BBHI Data Timept1 NPS.csv")

# Harmonize IDs
clean_id <- function(x) sub("^sub-", "", as.character(x))
raw_data$id <- clean_id(raw_data$id)
sfc_data_1 <- sfc_data_1 %>% rename(id = X)
sfc_data_2 <- sfc_data_2 %>% rename(id = X)
weighted <- weighted %>% rename(id = subject_id)
sfc_data_1$id <- clean_id(sfc_data_1$id)
sfc_data_2$id <- clean_id(sfc_data_2$id)
weighted$id <- clean_id(weighted$id)

# Baseline MMSE lookup
mmse_lookup <- full_join(
  bbhi_senior %>% transmute(id = clean_id(ID), mmse_senior = suppressWarnings(as.numeric(MMSE))),
  bbhi %>% transmute(id = clean_id(id), mmse_bbhi = suppressWarnings(as.numeric(w1_mmse))),
  by = "id"
) %>%
  mutate(w1_mmse = coalesce(mmse_senior, mmse_bbhi)) %>%
  dplyr::select(id, w1_mmse)

# Normalize weighted names to <variable>_<timepoint>
weighted <- weighted %>%
  rename_with(~ gsub("^w([12])_(.*)$", "\\2_\\1", .x)) %>%
  rename_with(~ gsub("^(.*)_tp([12])_(.*)$", "\\1_\\3_\\2", .x))

# Cohort inclusion: present in all files + complete key weighted summaries at both timepoints
key_summary_cols <- c(
  "sfc_weighted_mean_1", "sfc_weighted_mean_2"
)

missing_key_cols <- setdiff(key_summary_cols, names(weighted))
if (length(missing_key_cols) > 0) {
  stop(paste0("Missing key weighted summary columns: ", paste(missing_key_cols, collapse = ", ")))
}

ids_in_all_files <- Reduce(intersect, list(raw_data$id, sfc_data_1$id, sfc_data_2$id, weighted$id))
ids_complete_weighted <- weighted %>%
  filter(id %in% ids_in_all_files) %>%
  filter(if_all(all_of(key_summary_cols), ~ !is.na(.x))) %>%
  pull(id)
analysis_ids <- intersect(ids_in_all_files, ids_complete_weighted)

# Remove unsuffixed columns when paired _1/_2 versions exist
weighted <- weighted %>%
  {
    nm <- names(.)
    paired_base <- unique(sub("_(1|2)$", "", nm[grepl("_(1|2)$", nm)]))
    drop_unsuffixed <- setdiff(intersect(nm, paired_base), "id")
    dplyr::select(., -dplyr::any_of(drop_unsuffixed))
  }

# Build master wide df
weighted_cols <- setdiff(names(weighted), "id")
weighted_base_cols <- unique(sub("_(1|2)$", "", weighted_cols))

# Keep only SFC ROI cols not already supplied by weighted
sfc_cols_keep <- setdiff(setdiff(names(sfc_data_1), "id"), weighted_base_cols)

sfc1_wide <- sfc_data_1 %>%
  filter(id %in% analysis_ids) %>%
  dplyr::select(id, all_of(sfc_cols_keep)) %>%
  rename_with(~ paste0(.x, "_1"), -id)

sfc2_wide <- sfc_data_2 %>%
  filter(id %in% analysis_ids) %>%
  dplyr::select(id, all_of(sfc_cols_keep)) %>%
  rename_with(~ paste0(.x, "_2"), -id)

base_wide <- raw_data %>%
  filter(id %in% analysis_ids) %>%
  left_join(mmse_lookup, by = "id") %>%
  rename_with(~ sub("^w1_(.*)$", "\\1_1", .x), starts_with("w1_")) %>%
  rename_with(~ sub("^w2_(.*)$", "\\1_2", .x), starts_with("w2_"))

data_wide <- base_wide %>%
  dplyr::select(-any_of(weighted_cols)) %>%
  left_join(weighted %>% filter(id %in% analysis_ids), by = "id") %>%
  left_join(sfc1_wide, by = "id") %>%
  left_join(sfc2_wide, by = "id")

# Calculate annual change (slope) for SFC variables
calc_annual_change <- function(data, vars) {
  for (v in vars) {
    var1 <- paste0(v, "_1")
    var2 <- paste0(v, "_2")
    var_slope <- paste0(v, "_slope")
    if (all(c(var1, var2, "age_1", "age_2") %in% names(data))) {
      data[[var_slope]] <- (data[[var2]] - data[[var1]]) / (data$age_2 - data$age_1)
    }
  }
  return(data)
}

sfc_slope_vars <- c("sfc_weighted_mean", "sfc_hmod", "sfc_sensory", "sfc_dmn", "sfc_salience", "sfc_control")
data_wide <- calc_annual_change(data_wide, sfc_slope_vars)

# Build master long df
timepoint_cols <- names(data_wide)[grepl("_(1|2)$", names(data_wide))]
data_long <- data_wide %>%
  pivot_longer(
    cols = all_of(timepoint_cols),
    names_to = c(".value", "timepoint"),
    names_pattern = "(.*)_(1|2)$"
  ) %>%
  mutate(timepoint = as.integer(timepoint)) %>%
  group_by(id) %>%
  mutate(time = age - min(age, na.rm = TRUE)) %>%
  ungroup()

# Remove RAVLT_total = 0 from both dfs because missing data
data_long$ravlt_total <- na_if(data_long$ravlt_total, 0)
data_wide$ravlt_total_1 <- na_if(data_wide$ravlt_total_1, 0)
data_wide$ravlt_total_2 <- na_if(data_wide$ravlt_total_2, 0)

data_long$delayed_recall_raw <- na_if(data_long$delayed_recall_raw, 0)
data_wide$delayed_recall_raw_1 <- na_if(data_wide$delayed_recall_raw_1, 0)
data_wide$delayed_recall_raw_2 <- na_if(data_wide$delayed_recall_raw_2, 0)

# Plot variable distributions (categorical bar chart; numeric binned bar chart)
plot_bar_distributions <- function(df, vars, bins = 20) {
  missing_vars <- setdiff(vars, names(df))
  vars_present <- intersect(vars, names(df))

  if (length(vars_present) == 0) {
    stop("None of the requested variables were found in the provided data frame.")
  }

  make_plot <- function(v) {
    x <- df[[v]]

    if (is.numeric(x)) {
      ggplot(df, aes(x = .data[[v]])) +
        geom_histogram(bins = bins, fill = "#4C78A8", color = "white", alpha = 0.9) +
        labs(title = paste("Distribution of", v), x = v, y = "Count") +
        theme_classic(base_size = 11)
    } else {
      ggplot(df %>% drop_na(all_of(v)), aes(x = .data[[v]])) +
        geom_bar(fill = "#72B7B2", alpha = 0.9) +
        labs(title = paste("Distribution of", v), x = v, y = "Count") +
        theme_classic(base_size = 11) +
        theme(axis.text.x = element_text(angle = 30, hjust = 1))
    }
  }

  plots <- lapply(vars_present, make_plot)
  names(plots) <- vars_present

  for (p in plots) print(p)

  invisible(list(plots = plots, present = vars_present, missing = missing_vars))
}

# Variables used in the models/plots in this script (for quick distribution checks)
distribution_vars <- unique(c(
  "YoE", "age", "delayed_recall_raw", "ravlt_total", "sfc_weighted_mean", 
  "sfc_hmod", "sfc_sensory", "sfc_dmn", "sfc_salience"
))

# Plot all requested variables at once from data_long
distribution_plot_results <- plot_bar_distributions(
  df = data_long,
  vars = distribution_vars,
  bins = 20
)

#####################
# Descriptive stats #
#####################

# Cohort summary table
fmt_mean_sd <- function(x) {
  paste0(formatC(mean(x, na.rm = TRUE), format = "f", digits = 2), " ± ",
         formatC(sd(x, na.rm = TRUE), format = "f", digits = 2))
}
fmt_pct_female <- function(x) {
  paste0(formatC(mean(tolower(as.character(x)) == "female", na.rm = TRUE) * 100, format = "f", digits = 1), "%")
}
fmt_p <- function(p) ifelse(is.na(p), NA_character_, formatC(p, format = "f", digits = 4))

summary_df <- data_wide %>%
  mutate(
    group = if_else(superager_long == 1, "Superager", "Non-superager"),
    followup_duration = age_2 - age_1
  )

age_baseline_p <- tryCatch(t.test(age_1 ~ group, data = summary_df)$p.value, error = function(e) NA_real_)
followup_duration_p <- tryCatch(t.test(followup_duration ~ group, data = summary_df)$p.value, error = function(e) NA_real_)
yoe_p <- tryCatch(t.test(YoE ~ group, data = summary_df)$p.value, error = function(e) NA_real_)
mmse_p <- tryCatch(t.test(mmse_1 ~ group, data = summary_df)$p.value, error = function(e) NA_real_)
sex_p <- tryCatch(chisq.test(table(summary_df$group, tolower(as.character(summary_df$sex))))$p.value, error = function(e) NA_real_)

group_table <- tibble(
  Variable = c("N", "% female", "Age baseline", "Follow-up duration", "YoE", "MMSE"),
  `Whole cohort` = c(
    as.character(nrow(summary_df)),
    fmt_pct_female(summary_df$sex),
    fmt_mean_sd(summary_df$age_1),
    fmt_mean_sd(summary_df$followup_duration),
    fmt_mean_sd(summary_df$YoE),
    fmt_mean_sd(summary_df$mmse_1)
  ),
  `Non-superager` = c(
    as.character(sum(summary_df$group == "Non-superager", na.rm = TRUE)),
    fmt_pct_female(summary_df$sex[summary_df$group == "Non-superager"]),
    fmt_mean_sd(summary_df$age_1[summary_df$group == "Non-superager"]),
    fmt_mean_sd(summary_df$followup_duration[summary_df$group == "Non-superager"]),
    fmt_mean_sd(summary_df$YoE[summary_df$group == "Non-superager"]),
    fmt_mean_sd(summary_df$mmse_1[summary_df$group == "Non-superager"])
  ),
  `Superager` = c(
    as.character(sum(summary_df$group == "Superager", na.rm = TRUE)),
    fmt_pct_female(summary_df$sex[summary_df$group == "Superager"]),
    fmt_mean_sd(summary_df$age_1[summary_df$group == "Superager"]),
    fmt_mean_sd(summary_df$followup_duration[summary_df$group == "Superager"]),
    fmt_mean_sd(summary_df$YoE[summary_df$group == "Superager"]),
    fmt_mean_sd(summary_df$mmse_1[summary_df$group == "Superager"])
  ),
  p_value = c(
    NA_character_,
    fmt_p(sex_p),
    fmt_p(age_baseline_p),
    fmt_p(followup_duration_p),
    fmt_p(yoe_p),
    fmt_p(mmse_p)
  )
)
print(group_table)

#####################
# LME and LM models #
#####################

# Add a function for getting the stats from LME models
get_mixed_effects_stats <- function(formula, data, vars_priority) {
  # Fit the mixed-effects model
  model <- lmer(formula, data = data, REML = FALSE) # uses Full Maximum Likelihood 
  summary_model <- summary(model)
  coefficients <- summary_model$coefficients
  anova_results <- anova(model) # Get ANOVA results for p-values
  
  # Initialize the variable index
  variable_index <- NULL
  
  # Check for each variable in priority order and break on the first match
  for (var in vars_priority) {
    if (var %in% rownames(coefficients)) {
      variable_index <- which(rownames(coefficients) == var)
      break
    }
  }
  
  # Check if a relevant variable was found
  if (is.null(variable_index)) {
    stop("None of the priority variables are in the model.")
  }
  
  coef <- coefficients[variable_index, 1]
  ci <- confint(model, parm = rownames(coefficients)[variable_index])
  
  # Extract p-value from ANOVA table
  # Factor terms appear as e.g. "factor(x)1" in the coefficient table but
  # "factor(x)" in the ANOVA table — strip the level suffix after each ")"
  coef_name  <- rownames(coefficients)[variable_index]
  anova_name <- gsub("\\)([^:]*)", ")", coef_name)
  p_value <- anova_results[which(rownames(anova_results) == anova_name), "Pr(>F)"]
  
  # Compute marginal and conditional R²
  r2_vals <- r.squaredGLMM(model)
  marginal_r2 <- r2_vals[1]       # Variance explained by fixed effects only
  conditional_r2 <- r2_vals[2]    # Variance explained by fixed + random effects
  
  return(list(
    coef = coef,
    ci = ci,
    p_value = p_value,
    marginal_r2 = marginal_r2,
    conditional_r2 = conditional_r2
  ))
}

# Run FDR correction within each model group
run_group_fdr <- function(model_groups, method = "fdr") {
  extract_p <- function(x) {
    if (is.list(x) && "p_value" %in% names(x)) {
      return(as.numeric(x$p_value)[1])
    }
    if (is.numeric(x) && length(x) == 1) {
      return(as.numeric(x))
    }
    NA_real_
  }

  bind_rows(lapply(names(model_groups), function(group_name) {
    models <- model_groups[[group_name]]
    if (is.null(names(models))) {
      names(models) <- paste0("model_", seq_along(models))
    }
    p_vals <- vapply(models, extract_p, numeric(1))
    tibble(
      group = group_name,
      model = names(models),
      p_value = p_vals,
      p_fdr = p.adjust(p_vals, method = method)
    )
  })) %>%
    arrange(group, p_fdr)
}

# Add age_1 to the long df for use as a covariate in the mixed models
data_long <- data_long %>%
  left_join(data_wide %>% dplyr::select(id, age_1), by = "id")

###########################
# SFC by superager status #
###########################

# Get stats
vars <- c(
  "factor(superager_long)1"
)

sfc_superager_weighted <- get_mixed_effects_stats(scale(sfc_weighted_mean) ~ factor(superager_long) + scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data = data_long, vars)
sfc_superager_weighted
sfc_superager_hmod <- get_mixed_effects_stats(scale(sfc_hmod) ~ factor(superager_long) + scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data = data_long, vars)
sfc_superager_hmod
sfc_superager_sensory <- get_mixed_effects_stats(scale(sfc_sensory) ~ factor(superager_long) + scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data = data_long, vars)
sfc_superager_sensory
sfc_superager_dmn <- get_mixed_effects_stats(scale(sfc_dmn) ~ factor(superager_long) + scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data = data_long, vars)
sfc_superager_dmn
sfc_superager_salience <- get_mixed_effects_stats(scale(sfc_salience) ~ factor(superager_long) + scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data = data_long, vars)
sfc_superager_salience
sfc_superager_control <- get_mixed_effects_stats(scale(sfc_control) ~ factor(superager_long) + scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data = data_long, vars)
sfc_superager_control

vars_slope <- c("factor(superager_long)1:scale(time)")

sfc_superager_weighted_slope <- get_mixed_effects_stats(scale(sfc_weighted_mean) ~ factor(superager_long) * scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data = data_long, vars_slope)
sfc_superager_weighted_slope
sfc_superager_hmod_slope <- get_mixed_effects_stats(scale(sfc_hmod) ~ factor(superager_long) * scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data = data_long, vars_slope)
sfc_superager_hmod_slope
sfc_superager_sensory_slope <- get_mixed_effects_stats(scale(sfc_sensory) ~ factor(superager_long) * scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data = data_long, vars_slope)
sfc_superager_sensory_slope
sfc_superager_dmn_slope <- get_mixed_effects_stats(scale(sfc_dmn) ~ factor(superager_long) * scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data = data_long, vars_slope)
sfc_superager_dmn_slope
sfc_superager_salience_slope <- get_mixed_effects_stats(scale(sfc_salience) ~ factor(superager_long) * scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data = data_long, vars_slope)
sfc_superager_salience_slope
sfc_superager_control_slope <- get_mixed_effects_stats(scale(sfc_control) ~ factor(superager_long) * scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data = data_long, vars_slope)
sfc_superager_control_slope

##################
# SFC and memory #
##################

# Get stats
vars <- c(
  "scale(sfc_weighted_mean)",
  "scale(sfc_hmod)",
  "scale(sfc_sensory)",
  "scale(sfc_dmn)",
  "scale(sfc_salience)",
  "scale(sfc_control)"
)

ravlt_sfc_weighted <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_weighted_mean) + scale(age_1) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_weighted
ravlt_sfc_hmod <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_hmod) + scale(age_1) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_hmod
ravlt_sfc_sensory <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_sensory) + scale(age_1) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_sensory
ravlt_sfc_dmn <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_dmn) + scale(age_1) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_dmn
ravlt_sfc_salience <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_salience) + scale(age_1) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_salience
ravlt_sfc_control <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_control) + scale(age_1) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_control

vars <- c(
  "scale(sfc_weighted_mean):scale(time)",
  "scale(sfc_hmod):scale(time)",
  "scale(sfc_sensory):scale(time)",
  "scale(sfc_dmn):scale(time)",
  "scale(sfc_salience):scale(time)",
  "scale(sfc_control):scale(time)"
)

ravlt_sfc_weighted_slope <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_weighted_mean) * scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_weighted_slope
ravlt_sfc_hmod_slope <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_hmod) * scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_hmod_slope
ravlt_sfc_sensory_slope <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_sensory) * scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_sensory_slope
ravlt_sfc_dmn_slope <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_dmn) * scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_dmn_slope
ravlt_sfc_salience_slope <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_salience) * scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_salience_slope
ravlt_sfc_control_slope <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_control) * scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_control_slope

##################
# FDR correction #
##################

# Look at global SFC, heteromodal, DMN, SN, ECN, and sensory as a control 
ravlt_model_stats <- list(
  ravlt_sfc_hmod = ravlt_sfc_hmod,
  ravlt_sfc_weighted_mean = ravlt_sfc_weighted,
  ravlt_sfc_sensory = ravlt_sfc_sensory,
  ravlt_sfc_dmn = ravlt_sfc_dmn,
  ravlt_sfc_salience = ravlt_sfc_salience,
  ravlt_sfc_control = ravlt_sfc_control
)

sfc_model_stats <- list(
  sfc_weighted_mean = sfc_superager_weighted,
  sfc_hmod = sfc_superager_hmod,
  sfc_dmn = sfc_superager_dmn,
  sfc_salience = sfc_superager_salience,
  sfc_sensory = sfc_superager_sensory,
  sfc_control = sfc_superager_control
)

ravlt_slope_model_stats <- list(
  ravlt_sfc_hmod_slope = ravlt_sfc_hmod_slope,
  ravlt_sfc_weighted_mean_slope = ravlt_sfc_weighted_slope,
  ravlt_sfc_sensory_slope = ravlt_sfc_sensory_slope,
  ravlt_sfc_dmn_slope = ravlt_sfc_dmn_slope,
  ravlt_sfc_salience_slope = ravlt_sfc_salience_slope,
  ravlt_sfc_control_slope = ravlt_sfc_control_slope
)

sfc_slope_model_stats <- list(
  sfc_weighted_mean_slope = sfc_superager_weighted_slope,
  sfc_hmod_slope = sfc_superager_hmod_slope,
  sfc_dmn_slope = sfc_superager_dmn_slope,
  sfc_salience_slope = sfc_superager_salience_slope,
  sfc_sensory_slope = sfc_superager_sensory_slope,
  sfc_control_slope = sfc_superager_control_slope
)

fdr_results <- run_group_fdr(list(
  sfc_models = sfc_model_stats,
  ravlt_models = ravlt_model_stats,
  sfc_slope_models = sfc_slope_model_stats,
  ravlt_slope_models = ravlt_slope_model_stats
))

sfc_fdr_results <- fdr_results %>% filter(group == "sfc_models")
ravlt_fdr_results <- fdr_results %>% filter(group == "ravlt_models")
sfc_slope_fdr_results <- fdr_results %>% filter(group == "sfc_slope_models")
ravlt_slope_fdr_results <- fdr_results %>% filter(group == "ravlt_slope_models")

options(scipen = 999)
print(sfc_fdr_results)
print(ravlt_fdr_results)
print(sfc_slope_fdr_results)
print(ravlt_slope_fdr_results)
