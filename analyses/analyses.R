suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(lme4)
  library(lmerTest)
  library(ggplot2)
  library(MuMIn)
})

#############
# Constants #
#############

DATA_DIR <- "~/Documents/2023:2024/Data/Exported data"
ATLASES <- c("schaefer200", "schaefer400")
MAIN_ATLAS <- "schaefer200"  # Main analysis

# SFC networks short name 
NETWORKS <- c(
  global = "sfc_weighted_mean",
  hmod = "sfc_hmod",
  dmn = "sfc_dmn",
  control = "sfc_control",
  salience = "sfc_salience",
  sensory = "sfc_sensory"
)

#################
# Data cleaning #
#################

clean_id <- function(x) sub("^sub-", "", as.character(x))

# Load source data
raw_data <- read.csv(file.path(DATA_DIR, "superager_clean.csv"))
raw_data$id <- clean_id(raw_data$id)
demographics_table <- read.csv(file.path(DATA_DIR, "demographics_table.csv"), check.names = FALSE)  # From demographics_table.py

# Calculate annual change for SFC variables
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

# Build the long analysis df for one atlas
load_atlas_data <- function(raw_data, atlas) {
  read_atlas_csv <- function(prefix) read.csv(file.path(DATA_DIR, paste0(prefix, atlas, ".csv")))

  # Harmonize IDs 
  sfc_data_1 <- read_atlas_csv("all_sfc_data_ses-01_") %>% rename(id = X) %>% mutate(id = clean_id(id))
  sfc_data_2 <- read_atlas_csv("all_sfc_data_ses-02_") %>% rename(id = X) %>% mutate(id = clean_id(id))
  weighted <- read_atlas_csv("weighted_global_roi_averages_") %>%
    rename(id = subject_id) %>%
    mutate(id = clean_id(id)) %>%
    # Normalize weighted names to <variable>_<timepoint>
    rename_with(~ gsub("^w([12])_(.*)$", "\\2_\\1", .x)) %>%
    rename_with(~ gsub("^(.*)_tp([12])_(.*)$", "\\1_\\3_\\2", .x))

  # Cohort inclusion: present in all files + complete key weighted summaries at both timepoints
  key_summary_cols <- c("sfc_weighted_mean_1", "sfc_weighted_mean_2")

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
    rename_with(~ sub("^w1_(.*)$", "\\1_1", .x), starts_with("w1_")) %>%
    rename_with(~ sub("^w2_(.*)$", "\\1_2", .x), starts_with("w2_"))

  data_wide <- base_wide %>%
    dplyr::select(-any_of(weighted_cols)) %>%
    left_join(weighted %>% filter(id %in% analysis_ids), by = "id") %>%
    left_join(sfc1_wide, by = "id") %>%
    left_join(sfc2_wide, by = "id") %>%
    calc_annual_change(NETWORKS)

  # Build master long df, keeping baseline age as a covariate
  timepoint_cols <- names(data_wide)[grepl("_(1|2)$", names(data_wide))]
  data_wide %>%
    pivot_longer(
      cols = all_of(timepoint_cols),
      names_to = c(".value", "timepoint"),
      names_pattern = "(.*)_(1|2)$"
    ) %>%
    mutate(timepoint = as.integer(timepoint)) %>%
    group_by(id) %>%
    mutate(time = age - min(age, na.rm = TRUE)) %>%
    ungroup() %>%
    left_join(data_wide %>% dplyr::select(id, age_1), by = "id")
}

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

# Plot all listed variables at once from the main atlas data
atlas_data <- lapply(setNames(ATLASES, ATLASES), function(a) load_atlas_data(raw_data, a))
print(sapply(atlas_data, function(d) n_distinct(d$id)))  # Subjects per atlas
distribution_plot_results <- plot_bar_distributions(
  df = atlas_data[[MAIN_ATLAS]],
  vars = distribution_vars,
  bins = 20
)

#####################
# Descriptive stats #
#####################

# Cohort summary table 
print(demographics_table)

##############
# LME models #
##############

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

# Formula and term of interest per model type; {sfc} is replaced by each network column
MODEL_SPECS <- list(
  # SFC by superager status
  group = list(
    formula = "scale({sfc}) ~ factor(superager_long) + scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id)",
    term = "factor(superager_long)1"
  ),
  group_change = list(
    formula = "scale({sfc}) ~ factor(superager_long) * scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id)",
    term = "factor(superager_long)1:scale(time)"
  ),
  # SFC and memory
  memory = list(
    formula = "scale(delayed_recall_raw) ~ scale({sfc}) + scale(age_1) + scale(time) + sex + scale(YoE) + (1 | id)",
    term = "scale({sfc})"
  ),
  memory_change = list(
    formula = "scale(delayed_recall_raw) ~ scale({sfc}) * scale(time) + scale(age_1) + sex + scale(YoE) + (1 | id)",
    term = "scale({sfc}):scale(time)"
  )
)

# Fit every model type x network for one atlas, with FDR across networks within each model type
fit_lme_models <- function(data_long) {
  lapply(MODEL_SPECS, function(spec) {
    fits <- lapply(NETWORKS, function(sfc) {
      fill <- function(x) gsub("{sfc}", sfc, x, fixed = TRUE)
      get_mixed_effects_stats(as.formula(fill(spec$formula)), data_long, fill(spec$term))
    })
    p_fdr <- p.adjust(sapply(fits, `[[`, "p_value"), method = "fdr")
    Map(function(fit, p) c(fit, p_fdr = p), fits, p_fdr)
  })
}

# One row per model type x network, for printing
summarise_lme <- function(atlas_results) {
  bind_rows(lapply(names(atlas_results), function(type) {
    fits <- atlas_results[[type]]
    tibble(
      model = type,
      network = names(fits),
      coef = sapply(fits, `[[`, "coef"),
      ci_lo = sapply(fits, function(x) x$ci[1]),
      ci_hi = sapply(fits, function(x) x$ci[2]),
      p_value = sapply(fits, `[[`, "p_value"),
      p_fdr = sapply(fits, `[[`, "p_fdr")
    )
  }))
}

lme_results <- lapply(atlas_data, fit_lme_models)

options(scipen = 999)
for (atlas in ATLASES) {
  cat("\n", atlas, "\n")
  print(summarise_lme(lme_results[[atlas]]), n = Inf)
}

# For plot_sfc_em_cortex.py save the summary table
lme_summary <- bind_rows(lapply(ATLASES, function(a) mutate(summarise_lme(lme_results[[a]]), atlas = a, .before = 1)))
write.csv(lme_summary, file.path(DATA_DIR, "lme_results.csv"), row.names = FALSE)
