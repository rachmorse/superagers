suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(lme4)
  library(lmerTest)
  library(ggplot2)
  library(readxl)
  library(MuMIn)
  library(emmeans)
})

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
  "YoE", "age", "delayed_recall_raw", "ravlt_total", "pacc5", "pacc5_no_em",
  "sfc_weighted_mean", "fc_weighted_mean", "sc_weighted_mean",
  "sfc_hmod", "sfc_sensory"
))

# Plot all requested variables at once from data_long
distribution_plot_results <- plot_bar_distributions(
  df = data_long,
  vars = distribution_vars,
  bins = 20
)

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

#######
# Weighted summaries vs age
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

#####
# ROIs in all three models
cols <- c(
  "X7Networks_RH_DorsAttn_FEF_1",
  "X7Networks_RH_Default_PFCdPFCm_5",
  "X7Networks_RH_SalVentAttn_FrOperIns_1",
  "X7Networks_LH_Default_pCunPCC_2"
)

# Superager effect per ROI
results <- lapply(cols, function(feature_name) {
  dat <- data_long %>%
    dplyr::select(id, superager_long, sex, YoE, age, time, all_of(feature_name)) %>%
    mutate(Value = .data[[feature_name]]) %>%
    drop_na(Value, superager_long, time, age, sex, YoE, id)

  model <- lmer(Value ~ superager_long + time + age + sex + YoE + (1 | id), data = dat, REML = FALSE)
  coef_tab <- as.data.frame(summary(model)$coefficients)
  tibble(Feature = feature_name, beta = coef_tab["superager_long", "Estimate"], p = coef_tab["superager_long", "Pr(>|t|)"])
})

results_df <- bind_rows(results) %>%
  mutate(p_fdr = p.adjust(p, method = "fdr")) %>%
  arrange(p_fdr) %>%
  dplyr::select(Feature, beta, p, p_fdr)
print(results_df)

######
# Delayed recall association per ROI
memory_results <- lapply(cols, function(feature_name) {
  dat <- data_long %>%
    dplyr::select(id, delayed_recall_raw, sex, YoE, age, time, all_of(feature_name)) %>%
    mutate(Value = .data[[feature_name]]) %>%
    drop_na(delayed_recall_raw, Value, time, age, sex, YoE, id)

  model <- lmer(delayed_recall_raw ~ Value + time + age + sex + YoE + (1 | id), data = dat, REML = FALSE)
  coef_tab <- as.data.frame(summary(model)$coefficients)
  tibble(Feature = feature_name, beta = coef_tab["Value", "Estimate"], p = coef_tab["Value", "Pr(>|t|)"])
})

memory_results_df <- bind_rows(memory_results) %>%
  mutate(p_fdr = p.adjust(p, method = "fdr")) %>%
  arrange(p_fdr) %>%
  dplyr::select(Feature, beta, p, p_fdr)
print(memory_results_df)

# Plot FDR-significant ROIs by group
sig_results <- results_df %>%
  filter(p_fdr < 0.05) %>%
  arrange(p_fdr) %>%
  mutate(
    p_star = case_when(
      p < 0.001 ~ "***",
      p < 0.01 ~ "**",
      p < 0.05 ~ "*",
      TRUE ~ "ns"
    ),
    Feature_clean = recode(
      Feature,
      "X7Networks_RH_DorsAttn_FEF_1" = "R hemisphere dorsal attention frontal eye fields",
      "X7Networks_RH_Default_PFCdPFCm_5" = "R hemisphere default dorsal/medial prefrontal cortex",
      "X7Networks_LH_Default_pCunPCC_2" = "L hemisphere default precuneus posterior cingulate cortex",
      "X7Networks_RH_SalVentAttn_FrOperIns_1" = "R hemisphere salience/ventral attention frontal operculum insula"
    ),
    facet_label = Feature_clean
  )

sig_features <- sig_results %>% pull(Feature)

if (length(sig_features) > 0) {
  plot_df <- data_long %>%
    dplyr::select(id, superager_long, all_of(sig_features)) %>%
    pivot_longer(cols = all_of(sig_features), names_to = "Feature", values_to = "Value") %>%
    drop_na(Value, superager_long) %>%
    left_join(sig_results %>% dplyr::select(Feature, facet_label, p_star), by = "Feature") %>%
    mutate(
      superager_group = factor(superager_long, levels = c(0, 1), labels = c("non-superager", "superager")),
      facet_label = factor(facet_label, levels = sig_results$facet_label)
    )

  ann_df <- plot_df %>%
    group_by(facet_label) %>%
    summarise(
      y_min = min(Value, na.rm = TRUE),
      y_max = max(Value, na.rm = TRUE),
      p_star = first(p_star),
      .groups = "drop"
    ) %>%
    mutate(
      y_bar = y_max + 0.06 * (y_max - y_min),
      y_star = y_max + 0.10 * (y_max - y_min)
    )

  ggplot(plot_df, aes(x = superager_group, y = Value, fill = superager_group)) +
    geom_violin(trim = FALSE, alpha = 0.30, color = NA) +
    geom_boxplot(width = 0.18, outlier.shape = NA, alpha = 0.65, color = "black") +
    geom_jitter(width = 0.08, alpha = 0.22, size = 0.9, color = "black") +
    stat_summary(fun = mean, geom = "point", shape = 23, size = 2.5, fill = "white", color = "black") +
    geom_segment(data = ann_df, aes(x = 1, xend = 2, y = y_bar, yend = y_bar), inherit.aes = FALSE, linewidth = 0.45) +
    geom_text(data = ann_df, aes(x = 1.5, y = y_star, label = p_star), inherit.aes = FALSE, size = 5) +
    facet_wrap(~ facet_label, scales = "free_y", ncol = 3) +
    scale_fill_manual(values = c("non-superager" = "#0178bf", "superager" = "#FFAA00")) +
    labs(x = NULL, y = "SFC") +
    theme_classic(base_size = 12) +
    theme(legend.position = "none")
} else {
  cat("No FDR-significant features to plot.\n")
}

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
  p_value <- anova_results[which(rownames(anova_results) == rownames(coefficients)[variable_index]), "Pr(>F)"]
  r2 <- as.data.frame(VarCorr(model))$vcov[1]
  
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

#####
# Quick models directly on data_long
summary(lmer(scale(sfc_weighted_mean) ~ scale(superager_long) + scale(time) + scale(age) + sex + scale(YoE) + (1 | id), data = data_long, REML = FALSE))
summary(lmer(scale(sfc_hmod) ~ scale(superager_long) + scale(time) + scale(age) + sex + scale(YoE) + (1 | id), data = data_long, REML = FALSE))
summary(lmer(scale(sfc_sensory) ~ scale(superager_long) + scale(time) + scale(age) + sex + scale(YoE) + (1 | id), data = data_long, REML = FALSE))

summary(lmer(fc_weighted_mean ~ superager_long + time + age + sex + YoE + (1 | id), data = data_long, REML = FALSE))
summary(lmer(sc_weighted_mean ~ superager_long + time + age + sex + YoE + (1 | id), data = data_long, REML = FALSE))

# Get stats
vars <- c(
  "scale(superager_long)"
)

sfc_weighted_mean <- get_mixed_effects_stats(scale(sfc_weighted_mean) ~ scale(superager_long) + scale(time) + scale(age) + sex + scale(YoE) + (1 | id), data = data_long, vars)
sfc_weighted_mean
sfc_hmod <- get_mixed_effects_stats(scale(sfc_hmod) ~ scale(superager_long) + scale(time) + scale(age) + sex + scale(YoE) + (1 | id), data = data_long, vars)
sfc_hmod
sfc_sensory <- get_mixed_effects_stats(scale(sfc_sensory) ~ scale(superager_long) + scale(time) + scale(age) + sex + scale(YoE) + (1 | id), data = data_long, vars)
sfc_sensory

# SFC-only significance label from the adjusted mixed model
m_sfc <- lmer(sfc_hmod ~ superager_long + time + age + sex + YoE + (1 | id),
              data = data_long, REML = FALSE)
p_sfc <- as.data.frame(summary(m_sfc)$coefficients)["superager_long", "Pr(>|t|)"]
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

# SFC-only significance label from the adjusted mixed model
m_sfc <- lmer(sfc_hmod ~ superager_long + time + age + sex + YoE + (1 | id),
              data = data_long, REML = FALSE)
p_sfc <- as.data.frame(summary(m_sfc)$coefficients)["superager_long", "Pr(>|t|)"]
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

# PACC5: z-score each component within wave, then unweighted average
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

pacc5_tp1 <- pacc_tp1 %>%
  mutate(
    z_mmse = (mmse - tp1_ref$tp1_mean_mmse) / tp1_ref$tp1_sd_mmse,
    z_ravlt_total = (ravlt_total - tp1_ref$tp1_mean_ravlt_total) / tp1_ref$tp1_sd_ravlt_total,
    z_sdmt_total = (sdmt_total - tp1_ref$tp1_mean_sdmt_total) / tp1_ref$tp1_sd_sdmt_total,
    z_animals_total = (animals_total - tp1_ref$tp1_mean_animals_total) / tp1_ref$tp1_sd_animals_total
  ) %>%
  mutate(
    pacc5_1 = rowMeans(cbind(z_mmse, z_ravlt_total, z_sdmt_total, z_animals_total), na.rm = TRUE),
    pacc5_no_em_1 = rowMeans(cbind(z_mmse, z_sdmt_total, z_animals_total), na.rm = TRUE)
  ) %>%
  dplyr::select(id, pacc5_1, pacc5_no_em_1)

pacc5_tp2 <- pacc_tp2 %>%
  mutate(
    z_mmse = (mmse - tp1_ref$tp1_mean_mmse) / tp1_ref$tp1_sd_mmse,
    z_ravlt_total = (ravlt_total - tp1_ref$tp1_mean_ravlt_total) / tp1_ref$tp1_sd_ravlt_total,
    z_sdmt_total = (sdmt_total - tp1_ref$tp1_mean_sdmt_total) / tp1_ref$tp1_sd_sdmt_total,
    z_animals_total = (animals_total - tp1_ref$tp1_mean_animals_total) / tp1_ref$tp1_sd_animals_total
  ) %>%
  mutate(
    pacc5_2 = rowMeans(cbind(z_mmse, z_ravlt_total, z_sdmt_total, z_animals_total), na.rm = TRUE),
    pacc5_no_em_2 = rowMeans(cbind(z_mmse, z_sdmt_total, z_animals_total), na.rm = TRUE)
  ) %>%
  dplyr::select(id, pacc5_2, pacc5_no_em_2)

pacc5_wide <- full_join(pacc5_tp1, pacc5_tp2, by = "id") %>%
  group_by(id) %>%
  summarise(
    pacc5_1 = if (all(is.na(pacc5_1))) NA_real_ else mean(pacc5_1, na.rm = TRUE),
    pacc5_no_em_1 = if (all(is.na(pacc5_no_em_1))) NA_real_ else mean(pacc5_no_em_1, na.rm = TRUE),
    pacc5_2 = if (all(is.na(pacc5_2))) NA_real_ else mean(pacc5_2, na.rm = TRUE),
    pacc5_no_em_2 = if (all(is.na(pacc5_no_em_2))) NA_real_ else mean(pacc5_no_em_2, na.rm = TRUE),
    .groups = "drop"
  )

pacc5_long <- pacc5_wide %>%
  pivot_longer(
    cols = c(pacc5_1, pacc5_2, pacc5_no_em_1, pacc5_no_em_2),
    names_to = c(".value", "timepoint"),
    names_pattern = "(pacc5(?:_no_em)?)_(\\d)"
  ) %>%
  mutate(timepoint = as.integer(timepoint))

# Merge both PACC variants into master datasets
data_wide <- data_wide %>%
  left_join(pacc5_wide, by = "id")

data_long <- data_long %>%
  left_join(pacc5_long, by = c("id", "timepoint"))


# PACC model
m_pacc <- lmer(pacc5 ~ superager_long + time + age + sex + YoE + (1 | id), data = data_long, REML = FALSE)
summary(m_pacc)
summary(lmer(scale(pacc5) ~ scale(superager_long) * scale(age) + scale(time) + sex + scale(YoE) + (1 | id), data = data_long, REML = FALSE))

vars <- (
  "scale(superager_long):scale(age)"
)
superager_long_age <- get_mixed_effects_stats(scale(pacc5) ~ scale(superager_long) * scale(age) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
superager_long_age

vars <- (
  "scale(superager_long)"
)
superager_long <- get_mixed_effects_stats(scale(pacc5) ~ scale(superager_long) + scale(age) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
superager_long

plot_df <- data_long %>%
  dplyr::select(id, age, superager_long, pacc5) %>%
  tidyr::drop_na() %>%
  dplyr::mutate(
    superager_group = factor(superager_long, levels = c(0, 1),
                             labels = c("non-superager", "superager"))
  )

ggplot(plot_df, aes(x = age, y = pacc5)) +
  geom_line(aes(group = id), alpha = 0.18, linewidth = 0.3, color = "grey55") +
  geom_point(aes(color = superager_group), alpha = 0.55, size = 1.8) +
  geom_smooth(aes(color = superager_group), method = "lm", se = TRUE, linewidth = 1.2) +
  scale_color_manual(values = c("non-superager" = "#0178bf", "superager" = "#FFAA00")) +
  labs(x = "Age", y = "PACC", color = NULL) +
  theme_classic(base_size = 12)

# PACC model no EM
m_pacc_no_em <- lmer(pacc5_no_em ~ superager_long + time + age + sex + YoE + (1 | id), data = data_long, REML = FALSE)
summary(m_pacc_no_em)
summary(lmer(scale(pacc5_no_em) ~ scale(superager_long) * scale(age) + scale(time) + sex + scale(YoE) + (1 | id), data = data_long, REML = FALSE))

plot_df_no_em <- data_long %>%
  dplyr::select(id, age, superager_long, pacc5_no_em) %>%
  drop_na() %>%
  mutate(
    superager_group = factor(superager_long, levels = c(0, 1),
                             labels = c("non-superager", "superager"))
  )

ggplot(plot_df_no_em, aes(x = age, y = pacc5_no_em)) +
  geom_line(aes(group = id), alpha = 0.18, linewidth = 0.3, color = "grey55") +
  geom_point(aes(color = superager_group), alpha = 0.55, size = 1.8) +
  geom_smooth(aes(color = superager_group), method = "lm", se = TRUE, linewidth = 1.2) +
  scale_color_manual(values = c("non-superager" = "#0178bf", "superager" = "#FFAA00")) +
  labs(x = "Age", y = "PACC5 without EM", color = NULL) +
  theme_classic(base_size = 12)

# Look at sfc and mem
data_long$delayed_recall_raw_z <- scale(data_long$delayed_recall_raw)
data_long$sfc_hmod_z <- scale(data_long$sfc_hmod)

ravlt_sfc_hmod_z <- (lmer(delayed_recall_raw_z ~ sfc_hmod_z + scale(age) + scale(time) + sex + scale(YoE) + (1 | id), data = data_long, REML = FALSE))
summary(ravlt_sfc_hmod_z)
summary(lmer(delayed_recall_raw_z ~ scale(sfc_weighted_mean) + scale(age) + scale(time) + sex + scale(YoE) + (1 | id), data = data_long, REML = FALSE))

# Get stats
vars <- c(
  "sfc_hmod_z",
  "scale(sfc_weighted_mean)",
  "scale(sfc_sensory)"
)
delayed_recall_raw_z <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ sfc_hmod_z + scale(age) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
delayed_recall_raw_z
sfc_weighted_mean <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_weighted_mean) + scale(age) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
sfc_weighted_mean
sfc_sensory <- get_mixed_effects_stats(scale(delayed_recall_raw) ~ scale(sfc_sensory) + scale(age) + scale(time) + sex + scale(YoE) + (1 | id), data_long, vars)
sfc_sensory

m1 <- lmer(
  delayed_recall_raw_z ~ sfc_hmod_z + age + time + sex + YoE + (1 | id),
  data = data_long,
  REML = FALSE
)

emm_sfc <- emmeans(
  m1,
  ~ sfc_hmod_z,
  at = list(
    sfc_hmod_z = seq(
      min(data_long$sfc_hmod_z, na.rm = TRUE),
      max(data_long$sfc_hmod_z, na.rm = TRUE),
      length.out = 100
    )
  )
)

emm_df <- as.data.frame(emm_sfc)

ggplot(data_long, aes(x = sfc_hmod_z, y = delayed_recall_raw_z)) +
  geom_line(aes(group = id), alpha = 0.25, color = "grey50") +
  geom_point(alpha = 0.45, color = "grey35", size = 2) +
  geom_ribbon(
    data = emm_df,
    aes(x = sfc_hmod_z, ymin = lower.CL, ymax = upper.CL),
    inherit.aes = FALSE,
    fill = "#de8c8c",
    alpha = 0.18
  ) +
  geom_line(
    data = emm_df,
    aes(x = sfc_hmod_z, y = emmean),
    inherit.aes = FALSE,
    color = "#de8c8c",
    linewidth = 1.2
  ) +
  labs(
    x = "Structure-Function Coupling in Heteromodal Regions",
    y = "RAVLT Delayed"
  ) +
  theme_classic()
