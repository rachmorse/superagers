suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(lme4)
  library(lmerTest)
  library(ggplot2)
  library(patchwork)
  library(readxl)
  library(MuMIn)
  library(emmeans)
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

# Cohort inclusion: present in all files + complete key weighted summaries at both waves
key_summary_cols <- c(
  "sfc_weighted_mean_1", "sfc_weighted_mean_2",
  "fc_weighted_mean_1", "fc_weighted_mean_2",
  "sc_weighted_mean_1", "sc_weighted_mean_2"
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

# Remove RAVLT_total = 0 from both dfs
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
  "YoE", "age", "delayed_recall_raw", "ravlt_total", "cog_composite",
  "sfc_weighted_mean", "fc_weighted_mean", "sc_weighted_mean",
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

#############################
# Weighted summaries vs age #
#############################

weighted_plot_df <- data_long %>%
  dplyr::select(id, age, superager_long, sfc_weighted_mean, fc_weighted_mean, sc_weighted_mean) %>%
  pivot_longer(
    cols = c(sfc_weighted_mean, fc_weighted_mean, sc_weighted_mean),
    names_to = "metric",
    values_to = "value"
  ) %>%
  drop_na(age, value, superager_long) %>%
  mutate(
    superager_group = factor(superager_long, levels = c(0, 1), labels = c("non-superager", "superager")),
    metric = recode(
      metric,
      fc_weighted_mean = "Functional connectivity",
      sc_weighted_mean = "Structural connectivity",
      sfc_weighted_mean = "Structure function coupling"
    )
  )

ggplot(weighted_plot_df, aes(x = age, y = value, color = metric)) +
  geom_line(aes(group = id), alpha = 0.18, linewidth = 0.3) +
  geom_point(alpha = 0.35, size = 1.7) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1.2) +
  facet_wrap(~ metric, ncol = 3, scales = "free_y") +
  scale_color_manual(values = c(
    "Functional connectivity" = "#F8766D",
    "Structural connectivity" = "#00BA38",
    "Structure function coupling" = "#619CFF"
  )) +
  labs(x = "Age", y = "Weighted mean") +
  theme_gray() +
  guides(color = "none")

# Weighted summaries by superager status
ggplot(weighted_plot_df, aes(x = age, y = value, color = superager_group)) +
  geom_point(alpha = 0.30, size = 1.4) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1.1) +
  facet_wrap(~ metric, ncol = 3, scales = "free_y") +
  scale_color_manual(values = c("non-superager" = "#0178bf", "superager" = "#FFAA00")) +
  labs(x = "Age", y = "Weighted mean", color = NULL) +
  theme_gray()

# Weighted summaries within SFC networks
weighted_plot_networks <- data_long %>%
  dplyr::select(id, age, superager_long, sfc_dmn, sfc_salience, sfc_vis, sfc_sommot, sfc_hippocampus, sfc_hmod, sfc_sensory) %>%
  pivot_longer(
    cols = c(sfc_dmn, sfc_salience, sfc_vis, sfc_sommot, sfc_hippocampus, sfc_hmod, sfc_sensory),
    names_to = "metric",
    values_to = "value"
  ) %>%
  drop_na(age, value, superager_long) %>%
  mutate(
    superager_group = factor(superager_long, levels = c(0, 1), labels = c("non-superager", "superager")),
    metric = recode(
      metric,
      sfc_salience = "Salience SFC",
      sfc_vis = "Visual SFC",
      sfc_dmn = "DMN SFC",
      sfc_hippocampus = "HC SFC",
      sfc_sommot = "Motor SFC",
      sfc_hmod = "Heteromodal SFC",
      sfc_sensory = "Sensory SFC"
    )
  )

ggplot(weighted_plot_networks, aes(x = age, y = value, color = metric)) +
  geom_line(aes(group = id), alpha = 0.18, linewidth = 0.3) +
  geom_point(alpha = 0.35, size = 1.7) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1.2) +
  facet_wrap(~ metric, ncol = 3, scales = "free_y") +
  scale_color_manual(values = c(
    "Salience SFC" = "#F8766D",
    "Visual SFC" = "#00BA38",
    "DMN SFC" = "#619CFF",
    "HC SFC" = "#FE9EC7",
    "Motor SFC" = "#FFA95A",
    "Heteromodal SFC" = "#B7BDF7",
    "Sensory SFC" = "#0C7779"
  )) +
  labs(x = "Age", y = "Weighted mean") +
  theme_gray() +
  guides(color = "none")

# Weighted summaries by superager status
ggplot(weighted_plot_networks, aes(x = age, y = value, color = superager_group)) +
  geom_point(alpha = 0.30, size = 1.4) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1.1) +
  facet_wrap(~ metric, ncol = 3, scales = "free_y") +
  scale_color_manual(values = c("non-superager" = "#0178bf", "superager" = "#FFAA00")) +
  labs(x = "Age", y = "Weighted mean", color = NULL) +
  theme_gray()

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

# Another for LMs
get_regression_stats <- function(formula, data, vars_priority) {
  model <- lm(formula, data = data)
  summary_model <- summary(model)
  coefficients <- summary_model$coefficients
  
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
  ci <- confint(model)[variable_index, ]
  p_value <- coefficients[variable_index, 4]
  t_stat <- coefficients[variable_index, 3]
  adj_r2 <- summary_model$adj.r.squared
  r2 <- summary_model$r.squared
  
  return(list(coef = coef, ci = ci, p_value = p_value, t_stat = t_stat, adj_r2 = adj_r2, r2 = r2))
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

############################################
# Creating & analyzing cognitive composite #
############################################

# Cognitive composite: z-score each component relative to baseline, then unweighted average
bbhi_tp2 <- read.csv("~/Documents/2023:2024/Data/BBHI/BBHI Data Timept2 NPS.csv")
bbhi_senior_tp2 <- read.csv("~/Documents/2023:2024/Data/BBHI-Senior/Ministerio2024Wave2_DATA_2026-01-20_1245.csv")

bbhi_tp2$id <- clean_id(bbhi_tp2$id)
bbhi_senior_tp2$record_id_np_w2 <- clean_id(bbhi_senior_tp2$record_id_np_w2)

pacc_tp1 <- full_join(
  bbhi %>%
    transmute(
      id = clean_id(id),
      mmse = suppressWarnings(as.numeric(w1_mmse)),
      ravlt_total = suppressWarnings(as.numeric(w1_inm_recall_total_raw)),
      sdmt_total = suppressWarnings(as.numeric(w1_number_keys_raw)),
      animals_total = suppressWarnings(as.numeric(w1_sem_fluency_raw))
    ),
  bbhi_senior %>%
    transmute(
      id = clean_id(ID),
      mmse_senior = suppressWarnings(as.numeric(MMSE)),
      ravlt_total_senior = suppressWarnings(as.numeric(RAVLT_total_learn)),
      sdmt_total_senior = suppressWarnings(as.numeric(SDMT_correct)),
      animals_total_senior = suppressWarnings(as.numeric(Sem_fluency))
    ),
  by = "id"
) %>%
  mutate(
    # Harmonize SDMT to per-minute rate (BBHI: 2.0 min; BBHI-senior: 1.5 min)
    sdmt_total = sdmt_total / 2,
    sdmt_total_senior = sdmt_total_senior / 1.5,
    mmse = coalesce(mmse_senior, mmse),
    ravlt_total = coalesce(ravlt_total_senior, ravlt_total),
    sdmt_total = coalesce(sdmt_total_senior, sdmt_total),
    animals_total = coalesce(animals_total_senior, animals_total)
  ) %>%
  dplyr::select(id, mmse, ravlt_total, sdmt_total, animals_total)

pacc_tp2 <- full_join(
  bbhi_tp2 %>%
    transmute(
      id = clean_id(id),
      mmse = suppressWarnings(as.numeric(mmse)),
      ravlt_total = suppressWarnings(as.numeric(inm_recall_total_raw)),
      sdmt_total = suppressWarnings(as.numeric(number_keys_raw)),
      animals_total = suppressWarnings(as.numeric(sem_fluency_raw))
    ),
  bbhi_senior_tp2 %>%
    transmute(
      id = clean_id(record_id_np_w2),
      mmse_senior = suppressWarnings(as.numeric(mmse_w2)),
      ravlt_total_senior = suppressWarnings(as.numeric(ravlt_total_w2)),
      sdmt_total_senior = suppressWarnings(as.numeric(sdmt_w2)),
      animals_total_senior = suppressWarnings(as.numeric(animals_w2))
    ),
  by = "id"
) %>%
  mutate(
    # Harmonize SDMT to per-minute rate (BBHI: 2.0 min; BBHI-senior: 1.5 min)
    sdmt_total = sdmt_total / 2,
    sdmt_total_senior = sdmt_total_senior / 1.5,
    mmse = coalesce(mmse_senior, mmse),
    ravlt_total = coalesce(ravlt_total_senior, ravlt_total),
    sdmt_total = coalesce(sdmt_total_senior, sdmt_total),
    animals_total = coalesce(animals_total_senior, animals_total)
  ) %>%
  dplyr::select(id, mmse, ravlt_total, sdmt_total, animals_total)

# tp1 reference to use baseline mean and SD for tp2
tp1_ref <- pacc_tp1 %>%
  summarise(
    tp1_mean_mmse = mean(mmse, na.rm = TRUE),
    tp1_sd_mmse = sd(mmse, na.rm = TRUE),
    tp1_mean_ravlt_total = mean(ravlt_total, na.rm = TRUE),
    tp1_sd_ravlt_total = sd(ravlt_total, na.rm = TRUE),
    tp1_mean_sdmt_total = mean(sdmt_total, na.rm = TRUE),
    tp1_sd_sdmt_total = sd(sdmt_total, na.rm = TRUE),
    tp1_mean_animals_total = mean(animals_total, na.rm = TRUE),
    tp1_sd_animals_total = sd(animals_total, na.rm = TRUE)
  )

cog_composite_tp1 <- pacc_tp1 %>%
  mutate(
    z_mmse = (mmse - tp1_ref$tp1_mean_mmse) / tp1_ref$tp1_sd_mmse,
    z_ravlt_total = (ravlt_total - tp1_ref$tp1_mean_ravlt_total) / tp1_ref$tp1_sd_ravlt_total,
    z_sdmt_total = (sdmt_total - tp1_ref$tp1_mean_sdmt_total) / tp1_ref$tp1_sd_sdmt_total,
    z_animals_total = (animals_total - tp1_ref$tp1_mean_animals_total) / tp1_ref$tp1_sd_animals_total
  ) %>%
  mutate(
    cog_composite_1 = rowMeans(cbind(z_mmse, z_sdmt_total, z_animals_total), na.rm = TRUE)
  ) %>%
  dplyr::select(id, cog_composite_1)

cog_composite_tp2 <- pacc_tp2 %>%
  mutate(
    z_mmse = (mmse - tp1_ref$tp1_mean_mmse) / tp1_ref$tp1_sd_mmse,
    z_ravlt_total = (ravlt_total - tp1_ref$tp1_mean_ravlt_total) / tp1_ref$tp1_sd_ravlt_total,
    z_sdmt_total = (sdmt_total - tp1_ref$tp1_mean_sdmt_total) / tp1_ref$tp1_sd_sdmt_total,
    z_animals_total = (animals_total - tp1_ref$tp1_mean_animals_total) / tp1_ref$tp1_sd_animals_total
  ) %>%
  mutate(
    cog_composite_2 = rowMeans(cbind(z_mmse, z_sdmt_total, z_animals_total), na.rm = TRUE)
  ) %>%
  dplyr::select(id, cog_composite_2)

cog_composite_wide <- full_join(cog_composite_tp1, cog_composite_tp2, by = "id") %>%
  group_by(id) %>%
  summarise(
    cog_composite_1 = if (all(is.na(cog_composite_1))) NA_real_ else mean(cog_composite_1, na.rm = TRUE),
    cog_composite_2 = if (all(is.na(cog_composite_2))) NA_real_ else mean(cog_composite_2, na.rm = TRUE),
    .groups = "drop"
  )

cog_composite_long <- cog_composite_wide %>%
  pivot_longer(
    cols = c(cog_composite_1, cog_composite_2),
    names_to = c(".value", "timepoint"),
    names_pattern = "(cog_composite)_(\\d)"
  ) %>%
  mutate(timepoint = as.integer(timepoint))

# Merge cognitive composite into master datasets
data_wide <- data_wide %>%
  left_join(cog_composite_wide, by = "id")

data_long <- data_long %>%
  left_join(cog_composite_long, by = c("id", "timepoint"))

data_long <- data_long %>%
  left_join(data_wide %>% dplyr::select(id, age_1), by = "id")

# Cognitive composite models
vars <- c(
  "factor(superager_long)1:scale(time)"
)
cog_composite_superager_age <- get_mixed_effects_stats(scale(cog_composite) ~ factor(superager_long) * scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data_long, vars)
cog_composite_superager_age

vars <- c(
  "factor(superager_long)1"
)
cog_composite_superager <- get_mixed_effects_stats(scale(cog_composite) ~ factor(superager_long) + scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id), data_long, vars)
cog_composite_superager

plot_df_cog_composite <- data_long %>%
  dplyr::select(id, age, superager_long, cog_composite) %>%
  drop_na() %>%
  mutate(
    superager_group = factor(superager_long, levels = c(0, 1),
                             labels = c("non-superager", "superager"))
  )

p_cog <- ggplot(plot_df_cog_composite, aes(x = age, y = cog_composite)) +
  geom_line(aes(group = id), alpha = 0.18, linewidth = 0.3, color = "grey55") +
  geom_point(aes(color = superager_group), alpha = 0.55, size = 1.8) +
  geom_smooth(aes(color = superager_group), method = "lm", se = TRUE, linewidth = 1.2) +
  scale_color_manual(values = c("non-superager" = "#0178bf", "superager" = "#FFAA00")) +
  labs(x = "Age", y = "Cognitive Composite", color = NULL) +
  theme_classic(base_size = 12) +
  theme(legend.position = "none")

###################
# Analyzing RAVLT #
###################

# RAVLT total model
vars <- c(
  "factor(superager_long)1:scale(time)"
)

ravlt_total_superager_age <- get_mixed_effects_stats(
  scale(ravlt_total) ~ factor(superager_long) * scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id),
  data_long,
  vars
)

ravlt_total_superager_age

vars <- c(
  "factor(superager_long)1"
)

ravlt_total_superager <- get_mixed_effects_stats(
  scale(ravlt_total) ~ factor(superager_long) + scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id),
  data_long,
  vars
)

ravlt_total_superager

plot_df_ravlt_total <- data_long %>%
  dplyr::select(id, age, superager_long, ravlt_total) %>%
  drop_na() %>%
  mutate(
    superager_group = factor(
      superager_long,
      levels = c(0, 1),
      labels = c("non-superager", "superager")
    )
  )

p_ravlt <- ggplot(plot_df_ravlt_total, aes(x = age, y = ravlt_total)) +
  geom_line(aes(group = id), alpha = 0.18, linewidth = 0.3, color = "grey55") +
  geom_point(aes(color = superager_group), alpha = 0.55, size = 1.8) +
  geom_smooth(aes(color = superager_group), method = "lm", se = TRUE, linewidth = 1.2) +
  scale_color_manual(values = c("non-superager" = "#0178bf", "superager" = "#FFAA00")) +
  labs(x = "Age", y = "Episodic Memory", color = NULL) +
  theme_classic(base_size = 12)

fig2 <- (p_cog + p_ravlt) +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(size = 14, face = "bold"))

ggsave(
  filename = "~/superagers/figure2_cognition_memory.png",
  plot = fig2,
  width = 10, height = 4.5, dpi = 300
)

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

vars_slope <- c("factor(superager_long)1")
sfc_superager_weighted_slope <- get_regression_stats(scale(sfc_weighted_mean_slope) ~ factor(superager_long) + scale(age_1) + sex + scale(YoE), data = data_wide, vars_slope)
sfc_superager_weighted_slope
sfc_superager_hmod_slope <- get_regression_stats(scale(sfc_hmod_slope) ~ factor(superager_long) + scale(age_1) + sex + scale(YoE), data = data_wide, vars_slope)
sfc_superager_hmod_slope
sfc_superager_sensory_slope <- get_regression_stats(scale(sfc_sensory_slope) ~ factor(superager_long) + scale(age_1) + sex + scale(YoE), data = data_wide, vars_slope)
sfc_superager_sensory_slope
sfc_superager_dmn_slope <- get_regression_stats(scale(sfc_dmn_slope) ~ factor(superager_long) + scale(age_1) + sex + scale(YoE), data = data_wide, vars_slope)
sfc_superager_dmn_slope
sfc_superager_salience_slope <- get_regression_stats(scale(sfc_salience_slope) ~ factor(superager_long) + scale(age_1) + sex + scale(YoE), data = data_wide, vars_slope)
sfc_superager_salience_slope
sfc_superager_control_slope <- get_regression_stats(scale(sfc_control_slope) ~ factor(superager_long) + scale(age_1) + sex + scale(YoE), data = data_wide, vars_slope)
sfc_superager_control_slope

# SFC-only significance label from the adjusted mixed model
m_sfc <- lmer(sfc_hmod ~ factor(superager_long) + time + age_1 + sex + YoE + (1 | id),
              data = data_long, REML = FALSE)
p_sfc <- as.data.frame(summary(m_sfc)$coefficients)["factor(superager_long)1", "Pr(>|t|)"]
p_star <- dplyr::case_when(
  p_sfc < 0.001 ~ "***",
  p_sfc < 0.01 ~ "**",
  p_sfc < 0.05 ~ "*",
  TRUE ~ "ns"
)

plot_df <- data_long %>%
  dplyr::select(superager_long, sfc_hmod) %>%
  drop_na() %>%
  mutate(
    superager_group = factor(superager_long, levels = c(0, 1),
                             labels = c("non-superager", "superager"))
  )

y_min <- min(plot_df$sfc_hmod, na.rm = TRUE)
y_max <- max(plot_df$sfc_hmod, na.rm = TRUE)
y_bar <- y_max + 0.06 * (y_max - y_min)
y_star <- y_max + 0.10 * (y_max - y_min)

ggplot(plot_df, aes(x = superager_group, y = sfc_hmod, fill = superager_group)) +
  geom_violin(trim = FALSE, alpha = 0.30, color = NA) +
  geom_boxplot(width = 0.18, outlier.shape = NA, alpha = 0.65, color = "black") +
  geom_jitter(width = 0.08, alpha = 0.22, size = 0.9, color = "black") +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 2.5, fill = "white", color = "black") +
  annotate("segment", x = 1, xend = 2, y = y_bar, yend = y_bar, linewidth = 0.45) +
  annotate("text", x = 1.5, y = y_star, label = p_star, size = 5) +
  scale_fill_manual(values = c("non-superager" = "#0178bf", "superager" = "#FFAA00")) +
  labs(x = NULL, y = "SFC heteromodal mean") +
  theme_classic(base_size = 12) +
  theme(legend.position = "none")

##################
# SFC and memory #
##################
data_long$delayed_recall_raw_z <- scale(data_long$delayed_recall_raw)

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
  "scale(sfc_weighted_mean_slope)",
  "scale(sfc_hmod_slope)",
  "scale(sfc_sensory_slope)",
  "scale(sfc_dmn_slope)",
  "scale(sfc_salience_slope)",
  "scale(sfc_control_slope)"
)

ravlt_sfc_weighted_slope <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_weighted_mean_slope) + scale(age_1) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_weighted_slope
ravlt_sfc_hmod_slope <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_hmod_slope) + scale(age_1) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_hmod_slope
ravlt_sfc_sensory_slope <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_sensory_slope) + scale(age_1) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_sensory_slope
ravlt_sfc_dmn_slope <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_dmn_slope) + scale(age_1) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_dmn_slope
ravlt_sfc_salience_slope <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_salience_slope) + scale(age_1) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_salience_slope
ravlt_sfc_control_slope <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_control_slope) + scale(age_1) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
ravlt_sfc_control_slope

make_sfc_memory_panel <- function(sfc_var, x_label, data) {
  data$sfc_z <- scale(data[[sfc_var]])

  m <- lmer(
    delayed_recall_raw_z ~ sfc_z + age_1 + time + sex + YoE + (1 | id),
    data = data, REML = FALSE
  )

  emm_df <- as.data.frame(emmeans(
    m, ~ sfc_z,
    at = list(sfc_z = seq(min(data$sfc_z, na.rm = TRUE),
                          max(data$sfc_z, na.rm = TRUE),
                          length.out = 100))
  ))

  ggplot(data, aes(x = sfc_z, y = delayed_recall_raw_z)) +
    geom_line(aes(group = id), alpha = 0.25, color = "grey50") +
    geom_point(alpha = 0.45, color = "grey35", size = 2) +
    geom_ribbon(
      data = emm_df,
      aes(x = sfc_z, ymin = lower.CL, ymax = upper.CL),
      inherit.aes = FALSE,
      fill = "#de8c8c", alpha = 0.18
    ) +
    geom_line(
      data = emm_df,
      aes(x = sfc_z, y = emmean),
      inherit.aes = FALSE,
      color = "#de8c8c", linewidth = 1.2
    ) +
    labs(x = x_label, y = "RAVLT Delayed Recall") +
    theme_classic(base_size = 12)
}

p_sfc_global <- make_sfc_memory_panel("sfc_weighted_mean", "Global SFC",      data_long)
p_sfc_hmod   <- make_sfc_memory_panel("sfc_hmod",          "Heteromodal SFC", data_long)
p_sfc_dmn    <- make_sfc_memory_panel("sfc_dmn",           "DMN SFC",         data_long)

fig3 <- (p_sfc_global + p_sfc_hmod + p_sfc_dmn) +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(size = 14, face = "bold"))

ggsave(
  filename = "~/superagers/figure3_sfc_memory.png",
  plot = fig3,
  width = 14, height = 4.5, dpi = 300
)

##################
# FDR correction #
##################

# I think it makes sense to look at global SFC, heteromodal, DMN, SN, ECN, 
# and sensory as a control 
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

pacc_model_stats <- list(
  cog_composite_superager = cog_composite_superager,
  cog_composite_superager_age = cog_composite_superager_age,
  ravlt_total_superager = ravlt_total_superager,
  ravlt_total_superager_age = ravlt_total_superager_age
)

fdr_results <- run_group_fdr(list(
  cog_mem_models = pacc_model_stats,
  sfc_models = sfc_model_stats,
  ravlt_models = ravlt_model_stats,
  sfc_slope_models = sfc_slope_model_stats,
  ravlt_slope_models = ravlt_slope_model_stats
))

cog_mem_fdr_results <- fdr_results %>% filter(group == "cog_mem_models")
sfc_fdr_results <- fdr_results %>% filter(group == "sfc_models")
ravlt_fdr_results <- fdr_results %>% filter(group == "ravlt_models")
sfc_slope_fdr_results <- fdr_results %>% filter(group == "sfc_slope_models")
ravlt_slope_fdr_results <- fdr_results %>% filter(group == "ravlt_slope_models")

options(scipen = 999)
print(as.data.frame(cog_mem_fdr_results), row.names = FALSE)
print(sfc_fdr_results)
print(ravlt_fdr_results)
print(sfc_slope_fdr_results)
print(ravlt_slope_fdr_results)
