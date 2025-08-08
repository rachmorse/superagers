# Load necessary packages
library(dplyr)
library(tidyr)
library(stringr)
library(emmeans)
library(ggplot2)
library(broom.mixed)
library(lme4)
library(patchwork)
library(performance)

# Read and prepare the data
demo_data <- read.csv("~/Documents/2023:2024/Data/Exported data/clean_data_all.csv")
sfc_1 <- read.csv("~/Documents/2023:2024/Data/Exported data/grouped_SFC_ses-01.csv")
sfc_2 <- read.csv("~/Documents/2023:2024/Data/Exported data/grouped_SFC_ses-02.csv")

demo_data <- demo_data %>% 
  filter(fu_time > 1.8)

sfc_1 <- sfc_1 %>%
  rename_with(~ paste0(.x, "_1"), .cols = -id) 

sfc_2 <- sfc_2 %>%
  rename_with(~ paste0(.x, "_2"), .cols = -id) 

# Merge the clean dfs
merged_data <- demo_data %>% 
  full_join(sfc_1, by = "id") %>% 
  full_join(sfc_2, by = "id") 

# Adjusting for learning effects

# 1. Fit the simple regression of TP2 on TP1:
practice_mod <- lm(memory_2 ~ memory_1, data = merged_data)
summary(practice_mod)


# Extract the fitted intercept and slope:
beta0 <- coef(practice_mod)[["(Intercept)"]]
beta1 <- coef(practice_mod)[["memory_1"]]

# 2. Compute the residualized TP2 for each subject:
merged_data$memory_adj_2 <- with(merged_data,
                                 memory_2 - (beta0 + beta1 * memory_1)
)

# Quick check: the new adj scores should have mean ≈ 0
mean(merged_data$memory_adj_2, na.rm = TRUE)
hist(merged_data$memory_adj_2, breaks = 20,
     main = "Residualized TP2 Memory", xlab = "memory_adj_2")

hist(merged_data$memory_2, breaks = 20,
     main = "Unadjusted tp2 Memory", xlab = "memory_2")

merged_data <- merged_data %>%
  mutate(memory_adj_1 = memory_1)

ggplot(merged_data, aes(x = memory_2, y = memory_1)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = TRUE, color = "steelblue") +
  labs(
    x = "2",
    y = "1"
  ) +
  theme_minimal()

ggplot(merged_data, aes(x = memory_2, y = memory_adj_2)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = TRUE, color = "steelblue") +
  labs(
    x = "unadj",
    y = "adj"
  ) +
  theme_minimal()

# Pivot data from wide to long
long_data <- merged_data %>% 
  pivot_longer(
    cols         = matches("_(\\d+)$"),          # age_1, memory_1, …
    names_to     = c(".value", "timepoint"),     # .value puts var names into columns
    names_pattern= "(.*)_(\\d+)$"
  ) %>% 
  mutate(timepoint = as.integer(timepoint)) %>% 
  mutate(
    time = if_else(timepoint == 1, 0L, 1L)   # 1 when time==0, else 2
  ) %>% 
  mutate(id = as.numeric(gsub("sub-", "", id))) %>% 
  rename_with(
      ~ .x %>%
        str_extract("[^\\.]+\\.[^\\.]+$") %>%                   # Extract e.g., "Right.Hippocampus"
        str_to_lower() %>%                                      # Make it lower case
        str_replace_all("\\.", "_"),                            # Replace . with _
      .cols = starts_with("Subcortical")
  )

summary(lmer(scale(memory) ~ scale(X7Networks_LH_Limbic_OFC) + time + (1 | id) + YoE + sex, data = long_data))
summary(lmer(scale(memory) ~ factor(superager) * scale(age) + time + (1 | id) + YoE + sex, data = long_data))


summary(lmer(scale(memory) ~ scale(X7Networks_LH_SalVentAttn_PFCl) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
summary(lmer(scale(memory) ~ scale(X7Networks_LH_Cont_pCun) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
summary(lmer(scale(memory) ~ scale(X7Networks_LH_Limbic_TempPole) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))

summary(lmer(scale(X7Networks_LH_Cont_pCun) ~ factor(maintainer) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
summary(lmer(scale(X7Networks_LH_Cont_pCun) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))

summary(lmer(scale(X7Networks_LH_Limbic_TempPole) ~ factor(maintainer) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
summary(lmer(scale(X7Networks_LH_Limbic_TempPole) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))

summary(lmer(scale(right_accumbens) ~ factor(maintainer) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
summary(lmer(scale(right_accumbens) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))

summary(lmer(scale(X7Networks_LH_SalVentAttn_PFCl) ~ factor(maintainer) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
summary(lmer(scale(X7Networks_LH_SalVentAttn_PFCl) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))

summary(aov(scale(Subcortical.208..Right.Hippocampus_1) ~ factor(maintainer) + age_1 + YoE + sex, data = merged_data))
summary(aov(scale(Subcortical.208..Right.Hippocampus_2) ~ factor(maintainer) + age_2 + YoE + sex, data = merged_data))
summary(lmer(scale(right_hippocampus) ~ factor(maintainer) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))

summary(aov(scale(Subcortical.201..Left.Hippocampus_1) ~ factor(maintainer) + age_1 + YoE + sex, data = merged_data))
summary(aov(scale(Subcortical.201..Left.Hippocampus_2) ~ factor(maintainer) + age_2 + YoE + sex, data = merged_data))


long_data$cohort <- ifelse(long_data$id > 5000, "bbhi", "bbhi_senior")
superagers_data <- long_data %>% 
  filter(superager == 1)

superagers_data <- long_data %>% 
  filter(superager == 1)

nonsuperagers_data <- long_data %>% 
  filter(superager == 0)

long_data <- long_data %>% 
  mutate(
    sfc_Thalamus = left_thalamus + right_thalamus,
    sfc_Accumbens = left_accumbens + right_accumbens
  )

summary(lmer(scale(memory_adj) ~ scale(sfc_Limbic) + (1 | id) + time + age + YoE + sex + cohort, data = long_data))
summary(lmer(scale(memory_adj) ~ scale(sfc_Thalamus) + (1 | id) + time + age + YoE + sex + cohort, data = long_data))
summary(lmer(scale(memory_adj) ~ scale(sfc_Accumbens) + (1 | id) + time + age + YoE + sex + cohort, data = long_data))
summary(lmer(scale(memory_adj) ~ scale(sfc_Hippocampus) + (1 | id) + time + age + YoE + sex + cohort, data = long_data))

summary(lmer(scale(memory_adj) ~ scale(sfc_Limbic) + (1 | id) + time + age + YoE + sex + cohort, data = superagers_data))
summary(lmer(scale(memory_adj) ~ scale(sfc_Thalamus) + (1 | id) + time + age + YoE + sex + cohort, data = superagers_data))
summary(lmer(scale(memory_adj) ~ scale(sfc_Accumbens) + (1 | id) + time + age + YoE + sex + cohort, data = superagers_data))
summary(lmer(scale(memory_adj) ~ scale(sfc_Hippocampus) + (1 | id) + time + age + YoE + sex + cohort, data = superagers_data))
hc <- lmer(scale(memory_adj) ~ scale(sfc_Hippocampus) + (1 | id) + time + age + YoE + sex + cohort, data = superagers_data)
summary(hc)
r2_vals_hc <- r2(hc)
r2_vals_hc

summary(lmer(scale(sfc_Hippocampus) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))

ggplot(superagers_data, aes(x = sfc_Hippocampus, y = memory_adj, group = id, color = as.factor(id))) +
  geom_point() +
  geom_line() +
  labs(title = "Memory vs Hippocampus (colored by id)",
       x = "sfc_Hippocampus",
       y = "memory_adj") +
  theme(legend.position = "none") # Hide legend if too many subjects

summary(lmer(scale(memory_adj) ~ scale(sfc_Limbic) + (1 | id) + age + YoE + sex + cohort, data = nonsuperagers_data))
summary(lmer(scale(memory_adj) ~ scale(sfc_Thalamus) + (1 | id) + age + YoE + sex + cohort, data = nonsuperagers_data))
summary(lmer(scale(memory_adj) ~ scale(sfc_Accumbens) + (1 | id) + age + YoE + sex + cohort, data = nonsuperagers_data))
summary(lmer(scale(memory_adj) ~ scale(sfc_Hippocampus) + (1 | id) + age + YoE + sex + cohort, data = nonsuperagers_data))

summary(lmer(scale(memory_adj) ~ scale(X7Networks_RH_Cont_pCun) + (1 | id) + age + YoE + sex + cohort, data = long_data))
summary(lmer(scale(memory_adj) ~ scale(X7Networks_LH_Cont_pCun) + (1 | id) + age + YoE + sex + cohort, data = long_data))

summary(lm(scale(memory_2) ~ scale(X7Networks_RH_Cont_pCun_2) + age_2 + YoE + sex, data = merged_data))
summary(lmer(scale(left_hippocampus) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
summary(lmer(scale(right_hippocampus) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))

# These are the longitudinal ROIs
rois <- c(
  "X7Networks_LH_Cont_pCun",
  "X7Networks_LH_Limbic_TempPole",
  "X7Networks_LH_SalVentAttn_PFCl"
)

# rois <- c(
#   "X7Networks_LH_Cont_OFC", "X7Networks_LH_Cont_Temp", "X7Networks_LH_Cont_pCun",
#   "X7Networks_LH_Default_PHC", "X7Networks_LH_DorsAttn_PrCv", "X7Networks_LH_Limbic_OFC",
#   "X7Networks_LH_Limbic_TempPole", "X7Networks_LH_SalVentAttn_PFCl", "X7Networks_RH_Cont_Par",
#   "X7Networks_RH_Cont_Temp", "X7Networks_RH_Default_PFCv", "X7Networks_RH_Limbic_OFC",
#   "X7Networks_RH_SalVentAttn_PrC", "left_amygdala", "left_putamen",
#   "left_accumbens", "left_thalamus", "right_amygdala",
#   "right_pallidum", "right_accumbens", "right_thalamus"
# )

maintainer_p <- c()
superager_p <- c()

for (roi in rois) {
  # Maintainer:Time
  f_maint <- as.formula(
    paste0("scale(", roi, ") ~ factor(maintainer) * scale(time) + (1 | id) + age + YoE + sex")
  )
  m_maint <- lmer(f_maint, data = long_data)
  tidy_maint <- broom.mixed::tidy(m_maint)
  idx <- grep("factor\\(maintainer\\).*:.*scale\\(time\\)", tidy_maint$term)
  if (length(idx) > 0) {
    maintainer_p <- c(maintainer_p, tidy_maint$p.value[idx][1])
  } else {
    maintainer_p <- c(maintainer_p, NA)
  }
  
  # Superager:Time
  f_super <- as.formula(
    paste0("scale(", roi, ") ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex")
  )
  m_super <- lmer(f_super, data = long_data)
  tidy_super <- broom.mixed::tidy(m_super)
  idx <- grep("factor\\(superager\\).*:.*scale\\(time\\)", tidy_super$term)
  if (length(idx) > 0) {
    superager_p <- c(superager_p, tidy_super$p.value[idx][1])
  } else {
    superager_p <- c(superager_p, NA)
  }
}
maintainer_p_fdr <- p.adjust(maintainer_p, method = "fdr")
superager_p_fdr  <- p.adjust(superager_p,  method = "fdr")

# Combine into a data frame for easy viewing
results_sfc <- data.frame(
  ROI = rois,
  maintainer_p = maintainer_p,
  maintainer_p_fdr = maintainer_p_fdr,
  superager_p = superager_p,
  superager_p_fdr = superager_p_fdr
)
print(results_sfc)

######## Repeat for FC data

# Read and prepare the data
fc_1 <- read.csv("~/Documents/2023:2024/Data/Exported data/grouped_FC_ses-01.csv")
fc_2 <- read.csv("~/Documents/2023:2024/Data/Exported data/grouped_FC_ses-02.csv")

fc_1 <- fc_1 %>%
  rename_with(~ paste0(.x, "_1"), .cols = -id) 

fc_2 <- fc_2 %>%
  rename_with(~ paste0(.x, "_2"), .cols = -id) 

# Merge the clean dfs
fc_merged_data <- demo_data %>% 
  full_join(fc_1, by = "id") %>% 
  full_join(fc_2, by = "id") 

# Pivot data from wide to long
fc_long_data <- fc_merged_data %>% 
  pivot_longer(
    cols         = matches("_(\\d+)$"),          # age_1, memory_1, …
    names_to     = c(".value", "timepoint"),     # .value puts var names into columns
    names_pattern= "(.*)_(\\d+)$"
  ) %>% 
  mutate(timepoint = as.integer(timepoint)) %>% 
  mutate(
    time = if_else(timepoint == 1, 0L, 1L)   # 1 when time==0, else 2
  ) %>% 
  mutate(id = as.numeric(gsub("sub-", "", id))) %>% 
  rename_with(
    ~ .x %>%
      str_extract("[^\\.]+\\.[^\\.]+$") %>%                   # Extract e.g., "Right.Hippocampus"
      str_to_lower() %>%                                      # Make it lower case
      str_replace_all("\\.", "_"),                            # Replace . with _
    .cols = starts_with("Subcortical")
  )

# List of all outcome variables
rois <- c(
  "X7Networks_RH_Cont_pCun", 
  "X7Networks_RH_DorsAttn_Post", "X7Networks_RH_SalVentAttn_TempOccPar", 
  "X7Networks_LH_SalVentAttn_Med", "X7Networks_RH_SalVentAttn_Med"
)

maintainer_p <- c()
superager_p <- c()

for (roi in rois) {
  # Maintainer:Time
  f_maint <- as.formula(
    paste0("scale(", roi, ") ~ factor(maintainer) * scale(time) + (1 | id) + age + YoE + sex")
  )
  m_maint <- lmer(f_maint, data = fc_long_data)
  tidy_maint <- broom.mixed::tidy(m_maint)
  idx <- grep("factor\\(maintainer\\).*:.*scale\\(time\\)", tidy_maint$term)
  if (length(idx) > 0) {
    maintainer_p <- c(maintainer_p, tidy_maint$p.value[idx][1])
  } else {
    maintainer_p <- c(maintainer_p, NA)
  }
  
  # Superager:Time
  f_super <- as.formula(
    paste0("scale(", roi, ") ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex")
  )
  m_super <- lmer(f_super, data = long_data)
  tidy_super <- broom.mixed::tidy(m_super)
  idx <- grep("factor\\(superager\\).*:.*scale\\(time\\)", tidy_super$term)
  if (length(idx) > 0) {
    superager_p <- c(superager_p, tidy_super$p.value[idx][1])
  } else {
    superager_p <- c(superager_p, NA)
  }
}

maintainer_p_fdr <- p.adjust(maintainer_p, method = "fdr")
superager_p_fdr  <- p.adjust(superager_p,  method = "fdr")

# Combine into a data frame for easy viewing
results_fc <- data.frame(
  ROI = rois,
  maintainer_p = maintainer_p,
  maintainer_p_fdr = maintainer_p_fdr,
  superager_p = superager_p,
  superager_p_fdr = superager_p_fdr
)
print(results_fc)

# Center time variable if not already done
long_data$time_centered <- scale(long_data$time)

# Fit model
model <- lmer(
  scale(X7Networks_LH_Limbic_TempPole) ~ factor(superager) * time_centered + (1 | id) + age + YoE + sex,
  data = long_data
)

# Sequence for time
time_seq <- seq(
  from = min(long_data$time_centered, na.rm = TRUE),
  to = max(long_data$time_centered, na.rm = TRUE),
  length.out = 100
)

# Marginal means using emmeans
em_superager_time <- emmeans(
  model,
  specs = ~ factor(superager) * time_centered,
  at = list(time_centered = time_seq),
  reff = 0
)

# Predictions dataframe
predictions <- summary(em_superager_time) %>% as.data.frame()
colnames(predictions)

predictions <- summary(em_superager_time) %>%
  as.data.frame() %>%
  rename(
    predicted = emmean,
    lower_bound = lower.CL,
    upper_bound = upper.CL
  ) %>%
  mutate(time_centered = as.numeric(time_centered))

# Color palette
palette_superager <- c("0" = "#60B5FF", "1" = "pink") # or use actual levels if different

# Plot
ggplot() +
  geom_line(
    data = long_data,
    aes(x = time_centered, y = scale(X7Networks_LH_Limbic_TempPole), group = id),
    color = "lightgray", alpha = 0.5
  ) +
  geom_point(
    data = long_data,
    aes(x = time_centered, y = scale(X7Networks_LH_Limbic_TempPole)),
    color = "gray", size = 1
  ) +
  geom_ribbon(
    data = predictions,
    aes(x = time_centered, ymin = lower_bound, ymax = upper_bound, fill = factor(superager)),
    alpha = 0.3
  ) +
  geom_line(
    data = predictions,
    aes(x = time_centered, y = predicted, color = factor(superager)),
    size = 1.2
  ) +
  scale_color_manual(values = palette_superager, labels = c("Not superager", "superager")) +
  scale_fill_manual(values = palette_superager, labels = c("Not superager", "superager")) +
  labs(
    x = "Time (centered)",
    y = "X7Networks_LH_Limbic_TempPole (scaled)",
    color = "superager",
    fill = "superager"
  ) +
  theme_minimal() +
  theme(legend.position = c(0.8, 0.94))

# Increase is associated with better memory
summary(aov(scale(X7Networks_LH_Limbic_OFC_1) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))
summary(aov(scale(X7Networks_LH_Limbic_OFC_2) ~ factor(superager) + age_2 + YoE + sex, data = merged_data))
summary(lmer(scale(X7Networks_LH_Limbic_OFC) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))

summary(aov(scale(Subcortical.213..Right.Accumbens_1) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))
summary(aov(scale(Subcortical.213..Right.Accumbens_2) ~ factor(superager) + age_2 + YoE + sex, data = merged_data))
summary(lmer(scale(right_accumbens) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))

# Decrease is associated with better memory
summary(aov(scale(X7Networks_RH_Default_PFCv_1) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))
summary(aov(scale(X7Networks_RH_Default_PFCv_2) ~ factor(superager) + age_2 + YoE + sex, data = merged_data))


# Increase is associated with better memory
summary(aov(scale(X7Networks_LH_Default_PHC_1) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))
summary(aov(scale(X7Networks_LH_Default_PHC_2) ~ factor(superager) + age_2 + YoE + sex, data = merged_data))
summary(lmer(scale(X7Networks_LH_Default_PHC) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))

summary(aov(scale(Subcortical.204..Left.Putamen_1) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))
summary(aov(scale(Subcortical.204..Left.Putamen_2) ~ factor(superager) + age_2 + YoE + sex, data = merged_data))
summary(lmer(scale(left_putamen) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))

# Decrease is associated with better memory
summary(aov(scale(X7Networks_RH_Default_PFCv_1) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))
summary(aov(scale(X7Networks_RH_Default_PFCv_2) ~ factor(superager) + age_2 + YoE + sex, data = merged_data))

merged_data %>%
  group_by(superager) %>%
  summarise(mean_value = mean(X7Networks_RH_Default_PFCv_2, na.rm = TRUE))

summary(lmer(scale(X7Networks_RH_Default_PFCv) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
X7Networks_RH_Default_PFCv <- (lmer(scale(X7Networks_RH_Default_PFCv) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
r2_vals <- r2(X7Networks_RH_Default_PFCv)
r2_vals

summary(aov(scale(X7Networks_LH_Limbic_TempPole_1) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))
summary(aov(scale(X7Networks_LH_Limbic_TempPole_2) ~ factor(superager) + age_2 + YoE + sex, data = merged_data))

merged_data %>%
  group_by(superager) %>%
  summarise(mean_value = mean(X7Networks_LH_Limbic_TempPole_2, na.rm = TRUE))

summary(lmer(scale(X7Networks_LH_Limbic_TempPole) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
X7Networks_LH_Limbic_TempPole <- (lmer(scale(X7Networks_LH_Limbic_TempPole) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
r2_vals <- r2(X7Networks_LH_Limbic_TempPole)
r2_vals

superagers_data <- merged_data %>% 
  filter(superager == 1)

superagers_data_long <- long_data %>% 
  filter(superager == 1)

summary(lmer(scale(memory_adj) ~ scale(X7Networks_RH_Default_PFCv) + time + (1|id) + age + YoE + sex, data = superagers_data_long))
summary(lmer(scale(memory_adj) ~ scale(X7Networks_LH_Limbic_TempPole) + time + (1|id) + age + YoE + sex, data = superagers_data_long))

# List of all outcome variables
rois <- c(
  "X7Networks_LH_Default_PHC", 
  "left_putamen", 
  "X7Networks_LH_Limbic_TempPole", 
  "X7Networks_RH_Default_PFCv"
)

superager_p <- c()

for (roi in rois) {
  # Superager:Time
  f_super <- as.formula(
    paste0("scale(", roi, ") ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex")
  )
  m_super <- lmer(f_super, data = long_data)
  tidy_super <- broom.mixed::tidy(m_super)
  idx <- grep("factor\\(superager\\).", tidy_super$term)
  if (length(idx) > 0) {
    superager_p <- c(superager_p, tidy_super$p.value[idx][1])
  } else {
    superager_p <- c(superager_p, NA)
  }
}

superager_p_fdr  <- p.adjust(superager_p,  method = "fdr")

# Combine into a data frame for easy viewing
results_sfc_sa <- data.frame(
  ROI = rois,
  superager_p = superager_p,
  superager_p_fdr = superager_p_fdr
)
print(results_sfc_sa)

#PLOT
# ensure grouping variables are factors
long_data <- long_data %>%
  mutate(
    time      = factor(timepoint, levels = c(1, 2), labels = c("Baseline","Follow-up")),
    superager = factor(superager, labels = c("Non-superager","Superager"))
  )

pfc_plot <- ggplot(long_data, aes(x = time, y = X7Networks_RH_Default_PFCv, group = id, color = superager)) +
  # individual trajectories
  geom_line(alpha = 0.3, size = 0.5) +
  # group mean trajectories
  stat_summary(
    aes(group = superager),
    fun = mean, geom = "line", size = 1.5
  ) +
  stat_summary(
    aes(group = superager),
    fun.data = mean_se, geom = "errorbar", width = 0.1, size = 1
  ) +
  scale_x_discrete(
    name   = "Timepoint",
    expand = c(0.1, 0.1)      # <<< no extra padding on left/right
  ) +
  scale_y_continuous(
    limits = c(-0.1, 0.5)
  ) +
  scale_color_manual(values = c("Non-superager" = "steelblue", "Superager" = "tomato")) +
  labs(
    x     = "Timepoint",
    y     = "RH Default Ventral Prefrontal Cortex",
    color = "Group"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank(),
    plot.margin      = unit(c(1, 1, 1, 0.5), "cm")  # <<< trim left margin
  )

temppole_plot <- ggplot(long_data, aes(x = time, y = X7Networks_LH_Limbic_TempPole, group = id, color = superager)) +
  # individual trajectories
  geom_line(alpha = 0.3, size = 0.5) +
  # group mean trajectories
  stat_summary(
    aes(group = superager),
    fun = mean, geom = "line", size = 1.5
  ) +
  stat_summary(
    aes(group = superager),
    fun.data = mean_se, geom = "errorbar", width = 0.1, size = 1
  ) +
  scale_x_discrete(
    name   = "Timepoint",
    expand = c(0.1, 0.1)      # <<< no extra padding on left/right
  ) +
  scale_y_continuous(
    limits = c(-0.1, 0.5)
  ) +
  scale_color_manual(values = c("Non-superager" = "steelblue", "Superager" = "tomato")) +
  labs(
    x     = "Timepoint",
    y     = "LH Limbic Temporal Pole",
    color = "Group"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank(),
    plot.margin      = unit(c(1, 1, 1, 0.5), "cm")  # <<< trim left margin
  )

combined_plot <- pfc_plot + temppole_plot +
  plot_layout(ncol = 2, guides = "collect") &    # collect legends
  theme(legend.position = "bottom")              # put the shared legend below

combined_plot + 
  plot_annotation(
    title = "Trajectories of Structure-Function Coupling",
    theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5))
  )


long_data <- long_data %>%
  mutate(superager_factor = case_when(
    superager == 1 ~ "superager",
    superager == 0 ~ "non-superager",
    TRUE ~ "unknown"
  ))

# Scaled stats
long_data <- long_data %>% 
  filter(!is.na(memory_slopes))

model_scaled_mem <- lmer(scale(memory) ~ scale(age) * factor(superager) + (1 | id)  + sex + YoE, data = long_data)
summary(model_scaled_mem)
r2_vals <- r2(model_scaled_mem)
r2_vals

# 1) Fit the model
model_res_lme <- lmer(memory ~ age * factor(superager) + (1 | id)  + sex + YoE, data = long_data)
summary(model_res_lme)

# 2) Create a grid of ages for which we want predictions
age_seq <- seq(
  from = min(long_data$age, na.rm = TRUE),
  to   = max(long_data$age, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each superager_factor × age
em_2group <- emmeans(
  model_res_lme, 
  specs = ~ superager * age,
  at    = list(age = age_seq),
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
  mutate(age = as.numeric(age))  # Convert from factor if needed

# Define colors for each group
palette_1 <- c("Non-superager" = "#A35C7A", "Superager" = "#FFD65A")

# Create the two-group plot
fig1 <- ggplot() +
  geom_line(
    data = long_data, 
    aes(x = age, y = memory, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data, 
    aes(x = age, y = memory), 
    color = "gray", size = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = age, ymin = lower_bound, ymax = upper_bound, 
        fill = superager), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = age, y = predicted, color = superager), 
    size = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "Age", y = "Episodic Memory Composite", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))
fig1

# Demographics
summary_data <- merged_data %>%
  filter(!is.na(sfc_all_slopes))

summary_stats <- summary_data %>%
  summarise(
    N                  = n(),
    mean_age_base      = mean(age_1, na.rm = TRUE),
    min_age_base       = min(age_1,  na.rm = TRUE),
    max_age_base       = max(age_1,  na.rm = TRUE),
    pct_female         = mean(sex == "female", na.rm = TRUE) * 100,
    mean_followup      = mean(fu_time, na.rm = TRUE),
    min_followup       = min(fu_time,  na.rm = TRUE),
    max_followup       = max(fu_time,  na.rm = TRUE)
  )

print(summary_stats)

fu_stuff <- summary_data %>%
  select(id, age_1, age_2, fu_time)


# Long data 
merged_data <- merged_data %>% 
  mutate(
    right_hippocampus_slope    = (Subcortical.208..Right.Hippocampus_2 - Subcortical.208..Right.Hippocampus_1) / (age_2 - age_1),
    rh_cont_pcun_slope         = (X7Networks_RH_Cont_pCun_2 - X7Networks_RH_Cont_pCun_1) / (age_2 - age_1),
    left_hippocampus_slope     = (Subcortical.201..Left.Hippocampus_2 - Subcortical.201..Left.Hippocampus_1) / (age_2 - age_1),
    right_accumbens_slope      = (Subcortical.213..Right.Accumbens_2 - Subcortical.213..Right.Accumbens_1) / (age_2 - age_1),
    lh_cont_pcun_slope         = (X7Networks_LH_Cont_pCun_2 - X7Networks_LH_Cont_pCun_1) / (age_2 - age_1),
    lh_salventattn_pfcl_slope  = (X7Networks_LH_SalVentAttn_PFCl_2 - X7Networks_LH_SalVentAttn_PFCl_1) / (age_2 - age_1)
  )

summary(aov(scale(X7Networks_LH_SalVentAttn_PFCl_1) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))
summary(aov(scale(X7Networks_LH_SalVentAttn_PFCl_2) ~ factor(superager) + age_2 + YoE + sex, data = merged_data))
summary(lmer(scale(X7Networks_LH_SalVentAttn_PFCl) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
summary(aov(scale(lh_salventattn_pfcl_slope) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))

summary(aov(scale(Subcortical.201..Left.Hippocampus_1) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))
summary(aov(scale(Subcortical.201..Left.Hippocampus_2) ~ factor(superager) + age_2 + YoE + sex, data = merged_data))
summary(lmer(scale(left_hippocampus) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
summary(aov(scale(left_hippocampus_slope) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))

summary(aov(scale(X7Networks_RH_Cont_pCun_1) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))
summary(aov(scale(X7Networks_RH_Cont_pCun_2) ~ factor(superager) + age_2 + YoE + sex, data = merged_data))
summary(lmer(scale(X7Networks_RH_Cont_pCun) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
summary(aov(scale(rh_cont_pcun_slope) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))

summary(aov(scale(Subcortical.208..Right.Hippocampus_1) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))
summary(aov(scale(Subcortical.208..Right.Hippocampus_2) ~ factor(superager) + age_2 + YoE + sex, data = merged_data))
summary(lmer(scale(right_hippocampus) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
summary(aov(scale(right_hippocampus_slope) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))

summary(aov(scale(X7Networks_LH_Cont_pCun_1) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))
summary(aov(scale(X7Networks_LH_Cont_pCun_2) ~ factor(superager) + age_2 + YoE + sex, data = merged_data))
summary(lmer(scale(X7Networks_LH_Cont_pCun) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
summary(aov(scale(lh_cont_pcun_slope) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))

summary(aov(scale(Subcortical.213..Right.Accumbens_1) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))
summary(aov(scale(Subcortical.213..Right.Accumbens_2) ~ factor(superager) + age_2 + YoE + sex, data = merged_data))
summary(lmer(scale(right_accumbens) ~ factor(superager) * scale(time) + (1 | id) + age + YoE + sex, data = long_data))
summary(aov(scale(right_accumbens_slope) ~ factor(superager) + age_1 + YoE + sex, data = merged_data))

# Adjusting for learning effects

# 1. Fit the simple regression of TP2 on TP1:
practice_mod <- lm(memory_2 ~ memory_1, data = merged_data)
summary(practice_mod)


# Extract the fitted intercept and slope:
beta0 <- coef(practice_mod)[["(Intercept)"]]
beta1 <- coef(practice_mod)[["memory_1"]]

# 2. Compute the residualized TP2 for each subject:
merged_data$memory_adj_2 <- with(merged_data,
                               memory_2 - (beta0 + beta1 * memory_1)
)

# Quick check: the new adj scores should have mean ≈ 0
mean(merged_data$memory_adj_2, na.rm = TRUE)
hist(merged_data$memory_adj_2, breaks = 20,
     main = "Residualized TP2 Memory", xlab = "memory_adj_2")

hist(merged_data$memory_2, breaks = 20,
     main = "Unadjusted tp2 Memory", xlab = "memory_2")

merged_data <- merged_data %>%
  mutate(memory_adj_1 = memory_1)

ggplot(merged_data, aes(x = memory_2, y = memory_1)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = TRUE, color = "steelblue") +
  labs(
    x = "2",
    y = "1"
  ) +
  theme_minimal()

ggplot(merged_data, aes(x = memory_2, y = memory_adj_2)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = TRUE, color = "steelblue") +
  labs(
    x = "unadj",
    y = "adj"
  ) +
  theme_minimal()

long_data <- merged_data %>% 
  pivot_longer(
    cols         = matches("_(\\d+)$"),          # age_1, memory_1, …
    names_to     = c(".value", "timepoint"),     # .value puts var names into columns
    names_pattern= "(.*)_(\\d+)$"
  ) %>% 
  mutate(timepoint = as.integer(timepoint)) %>% 
  mutate(
    time = if_else(timepoint == 1, 0L, 1L)   # 1 when time==0, else 2
  ) %>% 
  mutate(id = as.numeric(gsub("sub-", "", id))) %>% 
  rename_with(
    ~ .x %>%
      str_extract("[^\\.]+\\.[^\\.]+$") %>%                   # Extract e.g., "Right.Hippocampus"
      str_to_lower() %>%                                      # Make it lower case
      str_replace_all("\\.", "_"),                            # Replace . with _
    .cols = starts_with("Subcortical")
  )

res_mem_slopes
summary(aov(scale(res_mem_slopes) ~ (superager) + age_1 + YoE + sex, data = merged_data))

summary(lmer(scale(memory) ~ scale(time) * (superager) + age + YoE + sex + (1 | id), data = long_data))
# 1) Fit the model
model_res_lme <- lmer(memory ~ time * (superager) + (1 | id)  + age + sex + YoE, data = long_data)
summary(model_res_lme)

# 2) Create a grid of ages for which we want predictions
time_seq <- seq(
  from = min(long_data$time, na.rm = TRUE),
  to   = max(long_data$time, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each supertimer_factor × time
em_2group <- emmeans(
  model_res_lme, 
  specs = ~ superager * time,
  at    = list(time = time_seq),
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
  mutate(time = as.numeric(time))  # Convert from factor if needed

predictions <- predictions %>%
  mutate(
    superager = factor(superager, levels = c(0,1),
                       labels = c("Control","Superager"))
  )

long_data <- long_data %>%
  mutate(
    superager = factor(superager, levels = c(0,1),
                       labels = c("Control","Superager"))
  )
# Define colors for each group
palette_1 <- c("Control" = "#A35C7A", "Superager" = "#FFD65A")

# Create the two-group plot
ggplot() +
  geom_line(
    data = long_data, 
    aes(x = time, y = memory, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data, 
    aes(x = time, y = memory), 
    color = "gray", size = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = time, ymin = lower_bound, ymax = upper_bound, 
        fill = superager), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = time, y = predicted, color = superager), 
    size = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "time", y = "Episodic memory Composite", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))

# BLOOP

keep_rois <- c(
  "X7Networks_LH_Limbic_OFC", 
  "X7Networks_RH_Cont_Cing", 
  "X7Networks_LH_Cont_Cing",
  "X7Networks_RH_Cont_PFCl",
  "X7Networks_RH_Cont_PFCmp",
  "X7Networks_RH_Cont_PFCv",
  "X7Networks_LH_Cont_PFCl",
  "X7Networks_LH_Cont_Temp",
  "X7Networks_RH_Cont_Temp",
  "X7Networks_LH_Default_PFC",
  "X7Networks_LH_Default_PHC",
  "X7Networks_LH_Default_Temp",
  "X7Networks_RH_Default_Temp",
  "X7Networks_LH_Default_Par",
  "X7Networks_RH_Default_Par",
  "X7Networks_LH_Default_pCunPCC",
  "X7Networks_RH_Default_pCunPCC",
  "Hippocampus",
  "Amygdala"
)


keep_rois <- c(
  "sfc_Default", 
  "func_within_Default", 
  "struct_within_Default", 
  "sfc_Hippocampus",
  "func_Hippocampus",
  "struct_Hippocampus"
)

# Loop through each ROI and run the model
model_summaries <- list()
for (roi in keep_rois) {
  formula_str <- paste0("scale(memory) ~ scale(", roi, ") * scale(time) + (1 | id) + age + YoE + sex")
  model <- lmer(as.formula(formula_str), data = long_data)
  cat("\n===== ROI:", roi, "=====\n")
  print(summary(model))
  model_summaries[[roi]] <- summary(model)
}


merged_data <- merged_data %>% 
  mutate(id = as.numeric(gsub("sub-", "", id)))
merged_data$cohort <- ifelse(merged_data$id > 5000, "bbhi", "bbhi_senior")

fit <- lm(memory_1 ~ age_1 + sex + YoE + cohort, data = merged_data)
merged_data <- merged_data %>%
  drop_na(memory_1) %>%
  mutate(
    memory_res_1 = residuals(fit)
  )
fit <- lm(memory_2 ~ age_2 + sex + YoE + cohort, data = merged_data)
merged_data <- merged_data %>%
  drop_na(memory_2) %>%
  mutate(
    memory_res_2 = residuals(fit)
  )

long_data <- merged_data %>% 
  pivot_longer(
    cols         = matches("_(\\d+)$"),          # age_1, memory_1, …
    names_to     = c(".value", "timepoint"),     # .value puts var names into columns
    names_pattern= "(.*)_(\\d+)$"
  ) %>% 
  mutate(timepoint = as.integer(timepoint)) %>% 
  mutate(
    time = if_else(timepoint == 1, 0L, 1L)   # 1 when time==0, else 2
  ) %>% 
  mutate(id = as.numeric(gsub("sub-", "", id))) %>% 
  rename_with(
    ~ .x %>%
      str_extract("[^\\.]+\\.[^\\.]+$") %>%                   # Extract e.g., "Right.Hippocampus"
      str_to_lower() %>%                                      # Make it lower case
      str_replace_all("\\.", "_"),                            # Replace . with _
    .cols = starts_with("Subcortical")
  )

long_data <- long_data %>%
  mutate(
    time      = factor(timepoint, levels = c(1, 2), labels = c("Baseline","Follow-up")),
    superager = factor(superager, labels = c("Non-superager","Superager"))
  )
ggplot(long_data, aes(x = time, y = memory_res, group = id, color = superager)) +
  geom_line(alpha = 0.3, linewidth = 0.5) +  # Use linewidth instead of size for lines
  stat_summary(
    aes(group = superager),
    fun = mean, geom = "line", linewidth = 1.5
  ) +
  stat_summary(
    aes(group = superager),
    fun.data = mean_se, geom = "errorbar", width = 0.1, linewidth = 1
  ) +
  scale_x_discrete(
    name   = "Timepoint",
    expand = c(0.1, 0.1)
  ) +
  scale_color_manual(values = c("Non-superager" = "steelblue", "Superager" = "tomato")) +
  labs(
    x     = "Timepoint",
    y     = "Memory Residuals",
    color = "Group"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank(),
    plot.margin      = unit(c(1, 1, 1, 0.5), "cm")
  )

# 1) Fit the model
long_data$memory_centered = scale(long_data$memory)
long_data$age_centered = scale(long_data$age)



model_res_lme <- lmer(memory ~ age * (superager) + (1 | id) + sex + YoE, data = long_data)
summary(model_res_lme)

# 2) Create a grid of ages for which we want predictions
age_seq <- seq(
  from = min(long_data$age, na.rm = TRUE),
  to   = max(long_data$age, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each supertimer_factor × age
em_2group <- emmeans(
  model_res_lme, 
  specs = ~ superager * age,
  at    = list(age = age_seq),
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
  mutate(age = as.numeric(age))   # Convert from factor if needed

# Define colors for each group
palette_1 <- c("Non-superager" = "tomato", "Superager" = "steelblue")

# Create the two-group plot
p1 <- ggplot() +
  geom_line(
    data = long_data, 
    aes(x = age, y = memory, group = id), 
    color = "lightgray", alpha = 0.5
  ) +
  geom_point(
    data = long_data, 
    aes(x = age, y = memory), 
    color = "gray", size = 1
  ) +
  geom_ribbon(
    data = predictions, 
    aes(x = age, ymin = lower_bound, ymax = upper_bound, fill = superager), 
    alpha = 0.3
  ) +
  geom_line(
    data = predictions, 
    aes(x = age, y = predicted, color = superager), 
    size = 1.2
  ) +
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +
  labs(x = "Age", y = "Episodic Memory Composite", color = "Group", fill = "Group") +
  theme_minimal(base_size = 22) +  # Makes most text bigger
  theme(
    legend.position.inside = c(0.8, 0.94), # keep your legend position
    axis.title = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 22),
    legend.title = element_text(size = 22, face = "bold"),
    legend.text = element_text(size = 22),
    plot.title = element_text(size = 22, face = "bold", hjust = 0.5)
  )

ggsave("SA vs nonSA.jpeg", plot = p1, width = 10, height = 6, dpi = 400, units = "in")
p1
