suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(lme4)
  library(lmerTest)
  library(ggplot2)
  library(readxl)
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

# Build master wide df
weighted_cols <- setdiff(names(weighted), "id")

base_wide <- raw_data %>%
  filter(id %in% analysis_ids) %>%
  left_join(mmse_lookup, by = "id") %>%
  rename_with(~ sub("^w1_(.*)$", "\\1_1", .x), starts_with("w1_")) %>%
  rename_with(~ sub("^w2_(.*)$", "\\1_2", .x), starts_with("w2_"))

data_wide <- base_wide %>%
  dplyr::select(-any_of(weighted_cols)) %>%
  left_join(weighted %>% filter(id %in% analysis_ids), by = "id")

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
  scale_color_manual(values = c("non-superager" = "#B39DDB", "superager" = "#F4D03F")) +
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
    scale_fill_manual(values = c("non-superager" = "#B39DDB", "superager" = "#F4D03F")) +
    labs(x = NULL, y = "SFC") +
    theme_classic(base_size = 12) +
    theme(legend.position = "none")
} else {
  cat("No FDR-significant features to plot.\n")
}

#####
# Quick models directly on data_long
summary(lmer(sfc_weighted_mean ~ superager_long + time + age + sex + YoE + (1 | id), data = data_long, REML = FALSE))
summary(lmer(sfc_weighted_mean ~ superager_long + time + age + sex + YoE + (1 | id), data = data_long, REML = FALSE))

summary(lmer(fc_weighted_mean ~ superager_long + time + age + sex + YoE + (1 | id), data = data_long, REML = FALSE))
summary(lmer(sc_weighted_mean ~ superager_long + time + age + sex + YoE + (1 | id), data = data_long, REML = FALSE))

# SFC-only significance label from the adjusted mixed model
m_sfc <- lmer(sfc_weighted_mean ~ superager_long + time + age + sex + YoE + (1 | id),
              data = data_long, REML = FALSE)
p_sfc <- as.data.frame(summary(m_sfc)$coefficients)["superager_long", "Pr(>|t|)"]
p_star <- dplyr::case_when(
  p_sfc < 0.001 ~ "***",
  p_sfc < 0.01 ~ "**",
  p_sfc < 0.05 ~ "*",
  TRUE ~ "ns"
)

plot_df <- data_long %>%
  dplyr::select(superager_long, sfc_weighted_mean) %>%
  tidyr::drop_na() %>%
  dplyr::mutate(
    superager_group = factor(superager_long, levels = c(0, 1),
                             labels = c("non-superager", "superager"))
  )

y_min <- min(plot_df$sfc_weighted_mean, na.rm = TRUE)
y_max <- max(plot_df$sfc_weighted_mean, na.rm = TRUE)
y_bar <- y_max + 0.06 * (y_max - y_min)
y_star <- y_max + 0.10 * (y_max - y_min)

ggplot(plot_df, aes(x = superager_group, y = sfc_weighted_mean, fill = superager_group)) +
  geom_violin(trim = FALSE, alpha = 0.30, color = NA) +
  geom_boxplot(width = 0.18, outlier.shape = NA, alpha = 0.65, color = "black") +
  geom_jitter(width = 0.08, alpha = 0.22, size = 0.9, color = "black") +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 2.5, fill = "white", color = "black") +
  geom_segment(aes(x = 1, xend = 2, y = y_bar, yend = y_bar), inherit.aes = FALSE, linewidth = 0.45) +
  geom_text(aes(x = 1.5, y = y_star, label = p_star), inherit.aes = FALSE, size = 5) +
  scale_fill_manual(values = c("non-superager" = "#B39DDB", "superager" = "#F4D03F")) +
  labs(x = NULL, y = "SFC weighted mean") +
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

pacc5_tp1 <- pacc_tp1 %>%
  mutate(
    z_mmse = as.numeric(scale(mmse)),
    z_ravlt_total = as.numeric(scale(ravlt_total)),
    z_sdmt_total = as.numeric(scale(sdmt_total)),
    z_animals_total = as.numeric(scale(animals_total))
  ) %>%
  mutate(
    pacc5_1 = rowMeans(cbind(z_mmse, z_ravlt_total, z_sdmt_total, z_animals_total), na.rm = TRUE)
  ) %>%
  dplyr::select(id, pacc5_1)

pacc5_tp2 <- pacc_tp2 %>%
  mutate(
    z_mmse = as.numeric(scale(mmse)),
    z_ravlt_total = as.numeric(scale(ravlt_total)),
    z_sdmt_total = as.numeric(scale(sdmt_total)),
    z_animals_total = as.numeric(scale(animals_total))
  ) %>%
  mutate(
    pacc5_2 = rowMeans(cbind(z_mmse, z_ravlt_total, z_sdmt_total, z_animals_total), na.rm = TRUE)
  ) %>%
  dplyr::select(id, pacc5_2)

pacc5_wide <- full_join(pacc5_tp1, pacc5_tp2, by = "id") %>%
  group_by(id) %>%
  summarise(
    pacc5_1 = if (all(is.na(pacc5_1))) NA_real_ else mean(pacc5_1, na.rm = TRUE),
    pacc5_2 = if (all(is.na(pacc5_2))) NA_real_ else mean(pacc5_2, na.rm = TRUE),
    .groups = "drop"
  )

pacc5_long <- pacc5_wide %>%
  pivot_longer(
    cols = c(pacc5_1, pacc5_2),
    names_to = "timepoint",
    names_pattern = "pacc5_(\\d)",
    values_to = "pacc5"
  ) %>%
  mutate(timepoint = as.integer(timepoint))

# Merge PACC5 into master datasets
data_wide <- data_wide %>%
  left_join(pacc5_wide, by = "id")

data_long <- data_long %>%
  left_join(pacc5_long, by = c("id", "timepoint"))

# PACC model
m_pacc <- lmer(pacc5 ~ superager_long + time + age + sex + YoE + (1 | id), data = data_long, REML = FALSE)
summary(m_pacc)
summary(lmer(pacc5 ~ superager_long * age + time + sex + YoE + (1 | id), data = data_long, REML = FALSE))

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
  scale_color_manual(values = c("non-superager" = "#B39DDB", "superager" = "#F4D03F")) +
  labs(x = "Age", y = "PACC5", color = NULL) +
  theme_classic(base_size = 12)

