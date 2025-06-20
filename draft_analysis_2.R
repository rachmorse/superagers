# Cleaning and prep ----

# Load necessary packages
library(dplyr)
library(tidyr)
library(stringr)
library(emmeans)
library(ggplot2)
library(broom.mixed)

# Read and prepare the data
data <- read.csv("~/Documents/2023:2024/Data/Exported data/clean_data_all.csv")
sys_1 <- read.csv("~/Documents/2023:2024/Data/Exported data/globalSyS_ses-01.csv")
sys_2 <- read.csv("~/Documents/2023:2024/Data/Exported data/globalSyS_ses-02.csv")

# Extract numeric id
data$id <- as.numeric(sub("sub-", "", data$id))

# Create cohort variable
data$cohort <- ifelse(data$id > 5000, "bbhi", "bbhi_senior")

sys_1 <- sys_1 %>% 
  rename(
    sys_v1_1 = gSyS_v1,
    sys_v2_1 = gSyS_v2,
    id = id_user
  ) %>% 
  mutate(id = as.numeric(gsub("sub-", "", id)))

sys_2 <- sys_2 %>% 
  rename(
    sys_v1_2 = gSyS_v1,
    sys_v2_2 = gSyS_v2,
    id = id_user
  ) %>% 
  mutate(id = as.numeric(gsub("sub-", "", id)))

# Merge the clean dfs
data <- data %>% 
  full_join(sys_1, by = "id") %>% 
  full_join(sys_2, by = "id") 

# Pivot data from wide to long
long_data <- data %>%
  pivot_longer(
    cols = matches("(.+)_(\\d+)$"),             
    names_to = c("base_var", "timepoint"),      
    names_pattern = "(.+)_(\\d+)$",             
    values_to = "value"                        
  ) %>%
  mutate(timepoint = as.numeric(timepoint)) %>% # convert timepoint to numeric
  pivot_wider(
    names_from = "base_var", 
    values_from = "value"
  )

# Create superager and maintainer chr versions
long_data$superager_chr <- ifelse(
  long_data$superager == 1,
  "superager",
  "non_superager"
)

long_data$superager_chr <- ifelse(
  long_data$superager == 1,
  "superager",
  "non_superager"
)

long_data$maintainer_chr <- ifelse(
  long_data$maintainer == 1,
  "maintainer",
  "decliner"
)

long_data$maintainer_chr <- ifelse(
  long_data$maintainer == 1,
  "maintainer",
  "decliner"
)

long_data$group <- ifelse(
  long_data$superager == 1 & long_data$maintainer == 0, "superager",
  ifelse(long_data$superager == 0 & long_data$maintainer == 1, "maintainer",
         NA
         )
)

# Make a time variable
long_data$time <- ifelse(long_data$timepoint == 1, 0, long_data$fu_time)
long_data$time <- ifelse(long_data$timepoint == 1, 0, long_data$fu_time)

########## OPTIONS FOR THE ANALYSIS ###########

# I think Im leaning toward running option one because 
# it asks do superagers have less age related decline in factor x and does this relate to memory 
# option two though is good too
# it asks does factor x lead to superagers having better memory 

# NOTE when the struct analysis is done, remember to replace long_data

######### OPTION ONE #############################
# One way to run the analysis here is to check first which variables are important in memory
summary(lmer(scale(memory) ~  scale(func_Hippocampus) + (1 | id)  + age + sex + YoE, data = long_data))

# Then if this is significant, run another model where age could be thought of as time while controlling for age
# in a model with just time everyone starts at 0 and it seems to not make as much sense to me 
# so this could be thought of as being a superager decreases the negative relationship between age and DMN FC
# eg superagers dont decrease in FC as much as non-superagers as they age
# this seems logical except in this cohort it seems that reduced FC is associated with better memory
# though this is driven by non-superagers, so it kind of seems like reduced FC is a compensation mechanism for non-superagers
summary(lmer(scale(func_all_Default) ~ superager_chr * scale(age) + (1 | id) + age + sex + YoE, data = long_data))

######### OPTION TWO #############################
# One possibility is to check a mediation analysis like the example below
# So to me this says do superagers have better memory because of their FC for example 
# but so far I have no significant examples of this to run the mediation analysis 
# superager -> memory
summary(lmer((memory) ~  superager_chr + (1 | id) + age + sex + YoE, data = long_data))

# then superager -> func/struct/sfc
summary(lmer((func_Hippocampus) ~ superager_chr + (1 | id) + age + sex + YoE, data = long_data))

# then func/struct/sfc -> memory
summary(lmer((memory) ~  (func_Hippocampus) + (1 | id)  + age + sex + YoE, data = long_data))

######### OPTION THREE #############################
# Could jump direction to an interaction model I just dont know if there would be criticism that superagers is defined using memory
# I suppose this would say something like an increase in FC is important for superagers and memory but increased FC do not affect memory in non-superager
# that also seems a but odd to interpret and I have no significant examples 
# the other thing though is that superagers are all meant to have good memory and looking at this says that superagers with decreased FC has bad memory
# which is also a bit odd
summary(lmer(scale(memory) ~  scale(func_Hippocampus) * superager_chr + (1 | id)  + age + sex + YoE, data = long_data))

######### OPTION FOUR #############################
# Could go to a more complicated interaction with age included but this is a bit harder to interpret
# An example from the data: in superagers with high struct HC connectivity there is a negative relationship between mem and age
# for superagers with low struct HC connectivity there is a positive relationship between mem and age
# for non-superagers there is no relationship between struct HC connectivity, memory, age
# meaning something like having low struct HC connectivity is important for preventing age related memory decline in superagers 
# and HC struct connectivty is not important for preventing age related memory decline in non-superagers
# but obviously this does not really make sense so thats fun 

summary(lmer(scale(memory) ~  scale(struct_Hippocampus) * scale(age) * superager_chr + (1 | id) + sex + YoE, data = long_data))

######################################
######### OPTION ONE #################
######################################
long_bbhi_senior <- long_data %>% 
  filter(cohort == "bbhi_senior")

# Filter to only subs with more than two values for each of the needed variables
long_data_fc <- long_data %>% 
  filter(!is.na(func_all_slopes) & !is.na(memory_slopes))

long_data_struct <- long_data %>% 
  filter(!is.na(struct_all_slopes) & !is.na(memory_slopes))

long_data_sfc <- long_data %>% 
  filter(!is.na(sfc_all_slopes) & !is.na(memory_slopes))

long_data_sys <- long_data %>%
  filter(!is.na(sys_v1)) %>% 
  group_by(id) %>%
  mutate(non_na_sys = sum(!is.na(sys_v1))) %>%
  filter(non_na_sys >= 2) %>%
  ungroup() %>%
  select(-non_na_sys)

long_data_fc_group <- long_data %>% 
  filter(!is.na(func_all_slopes) & !is.na(memory_slopes) & !is.na(group))

long_data_struct_group <- long_data %>% 
  filter(!is.na(struct_all_slopes) & !is.na(memory_slopes) & !is.na(group))

long_data_sfc_group <- long_data %>% 
  filter(!is.na(sfc_all_slopes) & !is.na(memory_slopes) & !is.na(group))

func_models <- list(
  func_all = lmer(scale(memory) ~ scale(func_all) + (1 | id) + age + cohort + sex + YoE, data = long_data_fc),
  func_within_DorsalAttention = lmer(scale(memory) ~ scale(func_within_DorsalAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data_fc),
  func_Hippocampus = lmer(scale(memory) ~ scale(func_Hippocampus) + (1 | id) + age + cohort + sex + YoE, data = long_data_fc),
  func_within_Subcortical = lmer(scale(memory) ~ scale(func_within_Subcortical) + (1 | id) + age + cohort + sex + YoE + cohort, data = long_data_fc),
  func_within_Default = lmer(scale(memory) ~ scale(func_within_Default) + (1 | id) + age + cohort + sex + YoE + cohort, data = long_data_fc),
  func_within_Frontoparietal = lmer(scale(memory) ~ scale(func_within_Frontoparietal) + (1 | id) + age + cohort + sex + YoE, data = long_data_fc),
  func_within_VentralAttention = lmer(scale(memory) ~ scale(func_within_VentralAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data_fc)
)

# Function to extract the p-value for the *first predictor* in each model
get_pval <- function(mod) {
  s <- summary(mod)
  as.numeric(coef(s)[2, "Pr(>|t|)"])
}

pvals <- sapply(func_models, get_pval)
pvals_fdr <- p.adjust(pvals, method = "fdr")

results_func <- data.frame(
  pval = pvals,
  pval_fdr = pvals_fdr,
  significant_fdr = pvals_fdr < 0.05
)
print(results_func)

struct_models <- list(
  struct_all = lmer(scale(memory) ~ scale(struct_all) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct),
  struct_Hippocampus = lmer(scale(memory) ~ scale(struct_Hippocampus) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct),
  struct_within_Subcortical = lmer(scale(memory) ~ scale(struct_within_Subcortical) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct),
  struct_within_Default = lmer(scale(memory) ~ scale(struct_within_Default) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct),
  struct_within_Frontoparietal = lmer(scale(memory) ~ scale(struct_within_Frontoparietal) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct),
  struct_within_VentralAttention = lmer(scale(memory) ~ scale(struct_within_VentralAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct),
  struct_within_DorsalAttention = lmer(scale(memory) ~ scale(struct_within_DorsalAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct)
)

pvals <- sapply(struct_models, get_pval)
pvals_fdr <- p.adjust(pvals, method = "fdr")

results_struct <- data.frame(
  pval = pvals,
  pval_fdr = pvals_fdr,
  significant_fdr = pvals_fdr < 0.05
)
print(results_struct)

# SFC Models
sfc_models <- list(
  sfc_all = lmer(scale(memory) ~ scale(sfc_all) + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc),
  sfc_Hippocampus = lmer(scale(memory) ~ scale(sfc_Hippocampus) + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc),
  sfc_Subcortical = lmer(scale(memory) ~ scale(sfc_Subcortical) + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc),
  sfc_Default = lmer(scale(memory) ~ scale(sfc_Default) + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc),
  sfc_Frontoparietal = lmer(scale(memory) ~ scale(sfc_Frontoparietal) + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc),
  sfc_VentralAttention = lmer(scale(memory) ~ scale(sfc_VentralAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc),
  sfc_DorsalAttention = lmer(scale(memory) ~ scale(sfc_DorsalAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc)
)

pvals <- sapply(sfc_models, get_pval)
pvals_fdr <- p.adjust(pvals, method = "fdr")

results_sfc <- data.frame(
  pval = pvals,
  pval_fdr = pvals_fdr,
  significant_fdr = pvals_fdr < 0.05
)
print(results_sfc)

## THEN ##
func_group_models <- list(
  func_sa <- lmer(scale(func_within_DorsalAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data_fc),
  func_maint <- lmer(scale(func_within_DorsalAttention) ~ maintainer_chr + (1 | id) + sex + age + YoE, data = long_data_fc),
  func_sa_maint <- lmer(scale(func_within_DorsalAttention) ~ group + (1 | id) + sex + age + YoE, data = long_data_fc_group)
)

pvals <- sapply(func_group_models, get_pval)
pvals_fdr <- p.adjust(pvals, method = "fdr")

results_group_func <- data.frame(
  pval = pvals,
  pval_fdr = pvals_fdr,
  significant_fdr = pvals_fdr < 0.05
)
print(results_group_func)

struct_group_models <- list(
  sfc_Hippocampus_superager = lmer(scale(sfc_Hippocampus) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data_sfc),
  sfc_VentralAttention_superager = lmer(scale(sfc_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data_sfc),
  sfc_Hippocampus_maintainer = lmer(scale(sfc_Hippocampus) ~ maintainer_chr + (1 | id) + sex + age + YoE, data = long_data_sfc),
  sfc_VentralAttention_maintainer = lmer(scale(sfc_VentralAttention) ~ maintainer_chr + (1 | id) + sex + age + YoE, data = long_data_sfc),
  sfc_Hippocampus_group = lmer(scale(sfc_Hippocampus) ~ group + (1 | id) + sex + age + YoE, data = long_data_sfc_group),
  sfc_VentralAttention_group = lmer(scale(sfc_VentralAttention) ~ group + (1 | id) + sex + age + YoE, data = long_data_sfc_group)
)

# Calculate and correct
group_pvals <- sapply(struct_group_models, get_pval)
group_pvals_fdr <- p.adjust(group_pvals, method = "fdr")

results_group_struct <- data.frame(
  pval = group_pvals,
  pval_fdr = group_pvals_fdr,
  significant_fdr = group_pvals_fdr < 0.05
)

print(results_group_struct)

#######
# 1) Fit the model

library(ggdist)
palette_1 <- c("maintainer" = "#60B5FF", "decliner" = "#FF9B17")

ggplot(long_data_sfc, aes(x = maintainer_chr, y = sfc_Hippocampus, fill = maintainer_chr)) +
  stat_halfeye(
    adjust = .5,
    width = .4,
    .width = 0,
    justification = -.4,
    point_colour = NA,
    alpha = 0.7
  ) +
  scale_fill_manual(values = palette_1) +
  scale_color_manual(values = palette_1) +
  geom_boxplot(width = .15, outlier.shape = NA, alpha = 0.5) +
  geom_jitter(width = .13, alpha = 0.3, size = 1) +
  theme_minimal() +
  labs(y = "SFC Hippocampus") +
  guides(fill = "none")

palette_1 <- c("maintainer" = "#60B5FF", "superager" = "pink")

ggplot(long_data_sfc_group, aes(x = group, y = sfc_Hippocampus, fill = group)) +
  stat_halfeye(
    adjust = .5,
    width = .4,
    .width = 0,
    justification = -.4,
    point_colour = NA,
    alpha = 0.7
  ) +
  scale_fill_manual(values = palette_1) +
  scale_color_manual(values = palette_1) +
  geom_boxplot(width = .15, outlier.shape = NA, alpha = 0.5) +
  geom_jitter(width = .13, alpha = 0.3, size = 1) +
  theme_minimal() +
  labs(y = "SFC Hippocampus") +
  guides(fill = "none")

ggplot(long_data_fc_group, aes(x = group, y = func_all, fill = group)) +
  stat_halfeye(
    adjust = .5,
    width = .4,
    .width = 0,
    justification = -.4,
    point_colour = NA,
    alpha = 0.7
  ) +
  scale_fill_manual(values = palette_1) +
  scale_color_manual(values = palette_1) +
  geom_boxplot(width = .15, outlier.shape = NA, alpha = 0.5) +
  geom_jitter(width = .13, alpha = 0.3, size = 1) +
  theme_minimal() +
  labs(y = "All FC") +
  guides(fill = "none")

model_memory_sfc <- lmer((memory) ~ (sfc_Hippocampus) + (1 | id) + age + sex + YoE + cohort, data = long_data_sfc)
summary(model_memory_sfc)

# 2) Create a grid of ages for which we want predictions
sfc_seq <- seq(
  from = min(long_data_sfc$sfc_Hippocampus, na.rm = TRUE),
  to   = max(long_data_sfc$sfc_Hippocampus, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for age only
em_overall <- emmeans(
  model_memory_sfc, 
  specs = ~ sfc_Hippocampus,
  at    = list(sfc_Hippocampus = sfc_seq),
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
  mutate(sfc_Hippocampus = as.numeric(sfc_Hippocampus))  
line_color <- "#129990" 

# Create the single trend line plot
ggplot() +
  geom_line(
    data = long_data_sfc, 
    aes(x = sfc_Hippocampus, y = memory, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data_sfc, 
    aes(x = sfc_Hippocampus, y = memory), 
    color = "gray", size = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = sfc_Hippocampus, ymin = lower_bound, ymax = upper_bound), 
    fill = line_color, alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = sfc_Hippocampus, y = predicted), 
    color = line_color, size = 1.5
  ) +  # Plot predicted line
  labs(x = "sfc_Hippocampus", y = "memory") +
  theme_minimal() +
  theme(legend.position = "none") 

##### Second plot
ggplot(long_data_sfc, aes(x = maintainer_chr, y = func_within_DorsalAttention, fill = maintainer_chr)) +
  stat_halfeye(
    adjust = .5,
    width = .4,       
    .width = 0,
    justification = -.4,
    point_colour = NA,
    alpha = 0.7
  ) +
  scale_fill_manual(values = palette_1) +
  scale_color_manual(values = palette_1) +
  geom_boxplot(width = .15, outlier.shape = NA, alpha = 0.5) +
  geom_jitter(width = .13, alpha = 0.3, size = 1) +
  theme_minimal() +
  labs(y = "Within Dorsal Attention FC") +
  guides(fill = "none")


model_memory_sfc <- lmer((memory) ~ (func_within_DorsalAttention) + (1 | id) + age + sex + YoE + cohort, data = long_data_sfc)
summary(model_memory_sfc)

# 2) Create a grid of ages for which we want predictions
sfc_seq <- seq(
  from = min(long_data_sfc$func_within_DorsalAttention, na.rm = TRUE),
  to   = max(long_data_sfc$func_within_DorsalAttention, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for age only
em_overall <- emmeans(
  model_memory_sfc, 
  specs = ~ func_within_DorsalAttention,
  at    = list(func_within_DorsalAttention = sfc_seq),
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
  mutate(func_within_DorsalAttention = as.numeric(func_within_DorsalAttention))  

# Create the single trend line plot
ggplot() +
  geom_line(
    data = long_data_sfc, 
    aes(x = func_within_DorsalAttention, y = memory, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data_sfc, 
    aes(x = func_within_DorsalAttention, y = memory), 
    color = "gray", size = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = func_within_DorsalAttention, ymin = lower_bound, ymax = upper_bound), 
    fill = line_color, alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = func_within_DorsalAttention, y = predicted), 
    color = line_color, size = 1.5
  ) +  # Plot predicted line
  labs(x = "func_within_DorsalAttention", y = "memory") +
  theme_minimal() +
  theme(legend.position = "none") 

# 2) Create a grid of ages for which we want predictions
sfc_seq <- seq(
  from = min(long_data_sfc$func_within_DorsalAttention, na.rm = TRUE),
  to   = max(long_data_sfc$func_within_DorsalAttention, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for age only
em_overall <- emmeans(
  model_memory_sfc, 
  specs = ~ func_within_DorsalAttention,
  at    = list(func_within_DorsalAttention = sfc_seq),
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
  mutate(func_within_DorsalAttention = as.numeric(func_within_DorsalAttention))  

# Create the single trend line plot
ggplot() +
  geom_line(
    data = long_data_sfc, 
    aes(x = func_within_DorsalAttention, y = memory, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data_sfc, 
    aes(x = func_within_DorsalAttention, y = memory), 
    color = "gray", size = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = func_within_DorsalAttention, ymin = lower_bound, ymax = upper_bound), 
    fill = line_color, alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = func_within_DorsalAttention, y = predicted), 
    color = line_color, size = 1.5
  ) +  # Plot predicted line
  labs(x = "func_within_DorsalAttention", y = "memory") +
  theme_minimal() +
  theme(legend.position = "none") 


######## Now look at where superagers / maintainers have different age-related changes in connectivity #####
# Function to extract the p-value for the *first predictor* in each model
get_pval_int <- function(mod) {
  s <- summary(mod)
  coef_table <- coef(s)
  last_row <- nrow(coef_table)
  as.numeric(coef_table[last_row, "Pr(>|t|)"])
}

# List of models to fit
func_maint_models <- list(
  func_all = lmer(scale(func_all) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_fc),
  func_within_DorsalAttention = lmer(scale(func_within_DorsalAttention) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_fc),
  func_Hippocampus = lmer(scale(func_Hippocampus) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_fc),
  func_within_Subcortical = lmer(scale(func_within_Subcortical) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_fc),
  func_within_Default = lmer(scale(func_within_Default) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_fc),
  func_within_Frontoparietal = lmer(scale(func_within_Frontoparietal) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_fc),
  func_within_VentralAttention = lmer(scale(func_within_VentralAttention) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_fc)
)

# Calculate p-values and apply FDR correction
func_maint_pvals <- sapply(func_maint_models, get_pval_int)
func_maint_pvals_fdr <- p.adjust(func_maint_pvals, method = "fdr")

results_func_maint <- data.frame(
  pval = func_maint_pvals,
  pval_fdr = func_maint_pvals_fdr,
  significant_fdr = func_maint_pvals_fdr < 0.05
)

print(results_func_maint)

# List of models to fit
func_sa_models <- list(
  func_all = lmer(scale(func_all) ~ superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_fc),
  func_within_DorsalAttention = lmer(scale(func_within_DorsalAttention) ~ superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_fc),
  func_Hippocampus = lmer(scale(func_Hippocampus) ~ superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_fc),
  func_within_Subcortical = lmer(scale(func_within_Subcortical) ~ superager_chr * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_fc),
  func_within_Default = lmer(scale(func_within_Default) ~ superager_chr * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_fc),
  func_within_Frontoparietal = lmer(scale(func_within_Frontoparietal) ~ superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_fc),
  func_within_VentralAttention = lmer(scale(func_within_VentralAttention) ~ superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_fc)
)

# Calculate p-values and apply FDR correction
func_sa_pvals <- sapply(func_sa_models, get_pval_int)
func_sa_pvals_fdr <- p.adjust(func_sa_pvals, method = "fdr")

results_func_sa <- data.frame(
  pval = func_sa_pvals,
  pval_fdr = func_sa_pvals_fdr,
  significant_fdr = func_sa_pvals_fdr < 0.05
)

print(results_func_sa)

# List of models to fit
func_sa_maint_models <- list(
  func_all = lmer(scale(func_all) ~ group * scale(age) + (1 | id) + sex + YoE, data = long_data_fc_group),
  func_within_DorsalAttention = lmer(scale(func_within_DorsalAttention) ~ group * scale(age) + (1 | id) + sex + YoE, data = long_data_fc_group),
  func_Hippocampus = lmer(scale(func_Hippocampus) ~ group * scale(age) + (1 | id) + sex + YoE, data = long_data_fc_group),
  func_within_Subcortical = lmer(scale(func_within_Subcortical) ~ group * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_fc_group),
  func_within_Default = lmer(scale(func_within_Default) ~ group * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_fc_group),
  func_within_Frontoparietal = lmer(scale(func_within_Frontoparietal) ~ group * scale(age) + (1 | id) + sex + YoE, data = long_data_fc_group),
  func_within_VentralAttention = lmer(scale(func_within_VentralAttention) ~ group * scale(age) + (1 | id) + sex + YoE, data = long_data_fc_group)
)

# Calculate p-values and apply FDR correction
func_sa_maint_pvals <- sapply(func_sa_maint_models, get_pval_int)
func_sa_maint_pvals_fdr <- p.adjust(func_sa_maint_pvals, method = "fdr")

results_func_sa_maint <- data.frame(
  pval = func_sa_maint_pvals,
  pval_fdr = func_sa_maint_pvals_fdr,
  significant_fdr = func_sa_maint_pvals_fdr < 0.05
)

print(results_func_sa_maint)

# Structural now 
# List of models to fit
struct_maint_models <- list(
  struct_all = lmer(scale(struct_all) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_struct),
  struct_within_DorsalAttention = lmer(scale(struct_within_DorsalAttention) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_struct),
  struct_Hippocampus = lmer(scale(struct_Hippocampus) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_struct),
  struct_within_Subcortical = lmer(scale(struct_within_Subcortical) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_struct),
  struct_within_Default = lmer(scale(struct_within_Default) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_struct),
  struct_within_Frontoparietal = lmer(scale(struct_within_Frontoparietal) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_struct),
  struct_within_VentralAttention = lmer(scale(struct_within_VentralAttention) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_struct)
)

# Calculate p-values and apply FDR correction
struct_maint_pvals <- sapply(struct_maint_models, get_pval_int)
struct_maint_pvals_fdr <- p.adjust(struct_maint_pvals, method = "fdr")

results_struct_maint <- data.frame(
  pval = struct_maint_pvals,
  pval_fdr = struct_maint_pvals_fdr,
  significant_fdr = struct_maint_pvals_fdr < 0.05
)

print(results_struct_maint)

# List of models to fit
struct_sa_models <- list(
  struct_all = lmer(scale(struct_all) ~ superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_struct),
  struct_within_DorsalAttention = lmer(scale(struct_within_DorsalAttention) ~ superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_struct),
  struct_Hippocampus = lmer(scale(struct_Hippocampus) ~ superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_struct),
  struct_within_Subcortical = lmer(scale(struct_within_Subcortical) ~ superager_chr * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_struct),
  struct_within_Default = lmer(scale(struct_within_Default) ~ superager_chr * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_struct),
  struct_within_Frontoparietal = lmer(scale(struct_within_Frontoparietal) ~ superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_struct),
  struct_within_VentralAttention = lmer(scale(struct_within_VentralAttention) ~ superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_struct)
)

# Calculate p-values and apply FDR correction
struct_sa_pvals <- sapply(struct_sa_models, get_pval_int)
struct_sa_pvals_fdr <- p.adjust(struct_sa_pvals, method = "fdr")

results_struct_sa <- data.frame(
  pval = struct_sa_pvals,
  pval_fdr = struct_sa_pvals_fdr,
  significant_fdr = struct_sa_pvals_fdr < 0.05
)

print(results_struct_sa)

# List of models to fit
struct_sa_maint_models <- list(
  struct_all = lmer(scale(struct_all) ~ group * scale(age) + (1 | id) + sex + YoE, data = long_data_struct_group),
  struct_within_DorsalAttention = lmer(scale(struct_within_DorsalAttention) ~ group * scale(age) + (1 | id) + sex + YoE, data = long_data_struct_group),
  struct_Hippocampus = lmer(scale(struct_Hippocampus) ~ group * scale(age) + (1 | id) + sex + YoE, data = long_data_struct_group),
  struct_within_Subcortical = lmer(scale(struct_within_Subcortical) ~ group * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_struct_group),
  struct_within_Default = lmer(scale(struct_within_Default) ~ group * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_struct_group),
  struct_within_Frontoparietal = lmer(scale(struct_within_Frontoparietal) ~ group * scale(age) + (1 | id) + sex + YoE, data = long_data_struct_group),
  struct_within_VentralAttention = lmer(scale(struct_within_VentralAttention) ~ group * scale(age) + (1 | id) + sex + YoE, data = long_data_struct_group)
)

# Calculate p-values and apply FDR correction
struct_sa_maint_pvals <- sapply(struct_sa_maint_models, get_pval_int)
struct_sa_maint_pvals_fdr <- p.adjust(struct_sa_maint_pvals, method = "fdr")

results_struct_sa_maint <- data.frame(
  pval = struct_sa_maint_pvals,
  pval_fdr = struct_sa_maint_pvals_fdr,
  significant_fdr = struct_sa_maint_pvals_fdr < 0.05
)

print(results_struct_sa_maint)

# SFC 
# List of models to fit
sfc_maint_models <- list(
  sfc_all = lmer(scale(sfc_all) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_sfc),
  sfc_DorsalAttention = lmer(scale(sfc_DorsalAttention) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_sfc),
  sfc_Hippocampus = lmer(scale(sfc_Hippocampus) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_sfc),
  sfc_Subcortical = lmer(scale(sfc_Subcortical) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_sfc),
  sfc_Default = lmer(scale(sfc_Default) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_sfc),
  sfc_Frontoparietal = lmer(scale(sfc_Frontoparietal) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_sfc),
  sfc_VentralAttention = lmer(scale(sfc_VentralAttention) ~ maintainer_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_sfc)
)

# Calculate p-values and apply FDR correction
sfc_maint_pvals <- sapply(sfc_maint_models, get_pval_int)
sfc_maint_pvals_fdr <- p.adjust(sfc_maint_pvals, method = "fdr")

results_sfc_maint <- data.frame(
  pval = sfc_maint_pvals,
  pval_fdr = sfc_maint_pvals_fdr,
  significant_fdr = sfc_maint_pvals_fdr < 0.05
)

print(results_sfc_maint)

# List of models to fit
sfc_sa_models <- list(
  sfc_all = lmer(scale(sfc_all) ~ superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_sfc),
  sfc_DorsalAttention = lmer(scale(sfc_DorsalAttention) ~ superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_sfc),
  sfc_Hippocampus = lmer(scale(sfc_Hippocampus) ~ superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_sfc),
  sfc_Subcortical = lmer(scale(sfc_Subcortical) ~ superager_chr * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_sfc),
  sfc_Default = lmer(scale(sfc_Default) ~ superager_chr * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_sfc),
  sfc_Frontoparietal = lmer(scale(sfc_Frontoparietal) ~ superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_sfc),
  sfc_VentralAttention = lmer(scale(sfc_VentralAttention) ~ superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data_sfc)
)

# Calculate p-values and apply FDR correction
sfc_sa_pvals <- sapply(sfc_sa_models, get_pval_int)
sfc_sa_pvals_fdr <- p.adjust(sfc_sa_pvals, method = "fdr")

results_sfc_sa <- data.frame(
  pval = sfc_sa_pvals,
  pval_fdr = sfc_sa_pvals_fdr,
  significant_fdr = sfc_sa_pvals_fdr < 0.05
)

print(results_sfc_sa)

# List of models to fit
sfc_sa_maint_models <- list(
  sfc_all = lmer(scale(sfc_all) ~ group * scale(age) + (1 | id) + sex + YoE, data = long_data_sfc_group),
  sfc_DorsalAttention = lmer(scale(sfc_DorsalAttention) ~ group * scale(age) + (1 | id) + sex + YoE, data = long_data_sfc_group),
  sfc_Hippocampus = lmer(scale(sfc_Hippocampus) ~ group * scale(age) + (1 | id) + sex + YoE, data = long_data_sfc_group),
  sfc_Subcortical = lmer(scale(sfc_Subcortical) ~ group * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_sfc_group),
  sfc_Default = lmer(scale(sfc_Default) ~ group * scale(age) + (1 | id) + sex + YoE + cohort, data = long_data_sfc_group),
  sfc_Frontoparietal = lmer(scale(sfc_Frontoparietal) ~ group * scale(age) + (1 | id) + sex + YoE, data = long_data_sfc_group),
  sfc_VentralAttention = lmer(scale(sfc_VentralAttention) ~ group * scale(age) + (1 | id) + sex + YoE, data = long_data_sfc_group)
)

# Calculate p-values and apply FDR correction
sfc_sa_maint_pvals <- sapply(sfc_sa_maint_models, get_pval_int)
sfc_sa_maint_pvals_fdr <- p.adjust(sfc_sa_maint_pvals, method = "fdr")

results_sfc_sa_maint <- data.frame(
  pval = sfc_sa_maint_pvals,
  pval_fdr = sfc_sa_maint_pvals_fdr,
  significant_fdr = sfc_sa_maint_pvals_fdr < 0.05
)

print(results_sfc_sa_maint)

# Center variables
long_data_fc$func_within_Default_centered = scale(long_data_fc$func_within_Default)
long_data_fc$age_centered = scale(long_data_fc$age)

# 1) Fit the model
func_dmn_sa <- lmer(func_within_Default_centered ~ age_centered * superager_chr + (1 | id) + sex + YoE, data = long_data_fc)
summary(func_dmn_sa)

age_centered_seq <- seq(
  from = min(long_data_fc$age_centered, na.rm = TRUE),
  to   = max(long_data_fc$age_centered, na.rm = TRUE),
  length.out = 100
)


# 3) Use emmeans to get marginal predictions for each superager_factor × sfc_all_centered
em_2group <- emmeans(
  func_dmn_sa, 
  specs = ~ factor(superager_chr) * age_centered,
  at    = list(age_centered = age_centered_seq),
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
palette_1 <- c("superager" = "#A35C7A", "non_superager" = "#FFD65A")

# Create the two-group plot
ggplot() +
  geom_line(
    data = long_data_fc, 
    aes(x = age_centered, y = func_within_Default_centered, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data_fc, 
    aes(x = age_centered, y = func_within_Default_centered), 
    color = "gray", size = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = age_centered, ymin = lower_bound, ymax = upper_bound, 
        fill = factor(superager_chr)), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = age_centered, y = predicted, color = factor(superager_chr)), 
    size = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "age", y = "func_within_Default_centered", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))

# Center variables
long_data_struct$struct_within_Frontoparietal_centered = scale(long_data_struct$struct_within_Frontoparietal)
long_data_struct$age_centered = scale(long_data_struct$age)

# 1) Fit the model
struct_fpn_maint <- lmer(struct_within_Frontoparietal_centered ~ age_centered * maintainer_chr + (1 | id) + sex + YoE , data = long_data_struct)
summary(struct_fpn_maint)

age_centered_seq <- seq(
  from = min(long_data_struct$age_centered, na.rm = TRUE),
  to   = max(long_data_struct$age_centered, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each superager_factor × sfc_all_centered
em_2group <- emmeans(
  struct_fpn_maint, 
  specs = ~ factor(maintainer_chr) * age_centered,
  at    = list(age_centered = age_centered_seq),
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
palette_1 <- c("maintainer" = "#60B5FF", "decliner" = "#FF9B17")

# Create the two-group plot
ggplot() +
  geom_line(
    data = long_data_struct, 
    aes(x = age_centered, y = struct_within_Frontoparietal_centered, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data_struct, 
    aes(x = age_centered, y = struct_within_Frontoparietal_centered), 
    color = "gray", size = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = age_centered, ymin = lower_bound, ymax = upper_bound, 
        fill = factor(maintainer_chr)), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = age_centered, y = predicted, color = factor(maintainer_chr)), 
    size = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "age", y = "struct_within_Frontoparietal_centered", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))

# Center variables
long_data_sfc_group$sfc_DorsalAttention_centered = scale(long_data_sfc_group$sfc_DorsalAttention)
long_data_sfc_group$age_centered = scale(long_data_sfc_group$age)

# 1) Fit the model
sfc_da <- lmer(sfc_DorsalAttention_centered ~ age_centered * group + (1 | id) + sex + YoE , data = long_data_sfc_group)
summary(sfc_da)

age_centered_seq <- seq(
  from = min(long_data_sfc_group$age_centered, na.rm = TRUE),
  to   = max(long_data_sfc_group$age_centered, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each superager_factor × sfc_all_centered
em_2group <- emmeans(
  sfc_da, 
  specs = ~ factor(group) * age_centered,
  at    = list(age_centered = age_centered_seq),
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
palette_1 <- c("maintainer" = "#60B5FF", "superager" = "pink")

# Create the two-group plot
ggplot() +
  geom_line(
    data = long_data_sfc_group, 
    aes(x = age_centered, y = sfc_DorsalAttention_centered, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data_sfc_group, 
    aes(x = age_centered, y = sfc_DorsalAttention_centered), 
    color = "gray", size = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = age_centered, ymin = lower_bound, ymax = upper_bound, 
        fill = factor(group)), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = age_centered, y = predicted, color = factor(group)), 
    size = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "age", y = "sfc_DorsalAttention_centered", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))

#### Memory and variables by superager vs maintainer
summary(lmer(scale(memory) ~  scale(func_all) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_fc_group))
summary(lmer(scale(memory) ~  scale(func_within_DorsalAttention) * group + (1 | id)  + age + cohort + sex + YoE, data = long_data_fc_group))
summary(lmer(scale(memory) ~  scale(func_all_VentralAttention) * group + (1 | id)  + age + cohort + sex + YoE, data = long_data_fc_group))
summary(lmer(scale(memory) ~  scale(func_all_DorsalAttention) * group + (1 | id)  + age + cohort + sex + YoE, data = long_data_fc_group))
summary(lmer(scale(memory) ~  scale(func_Hippocampus) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_fc_group))
summary(lmer(scale(memory) ~  scale(func_within_Subcortical) * group + (1 | id) + age + cohort + sex + YoE + cohort, data = long_data_fc_group))
summary(lmer(scale(memory) ~  scale(func_within_Default) * group + (1 | id) + age + cohort + sex + YoE + cohort, data = long_data_fc_group))
summary(lmer(scale(memory) ~  scale(func_within_Frontoparietal) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_fc_group))
summary(lmer(scale(memory) ~  scale(func_within_VentralAttention) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_fc_group))
summary(lmer(scale(memory) ~  scale(func_all_Subcortical) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_fc_group))
summary(lmer(scale(memory) ~  scale(func_all_Default) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_fc_group))
summary(lmer(scale(memory) ~  scale(func_all_Frontoparietal) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_fc_group))

summary(lmer(scale(memory) ~  scale(struct_all) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(memory) ~  scale(struct_Hippocampus) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(memory) ~  scale(struct_within_Subcortical) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(memory) ~  scale(struct_within_Default) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(memory) ~  scale(struct_within_Frontoparietal) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(memory) ~  scale(struct_within_VentralAttention) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(memory) ~  scale(struct_within_DorsalAttention) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(memory) ~  scale(struct_all_Subcortical) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(memory) ~  scale(struct_all_Default) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(memory) ~  scale(struct_all_Frontoparietal) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(memory) ~  scale(struct_all_VentralAttention) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(memory) ~  scale(struct_all_DorsalAttention) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_struct_group))

summary(lmer(scale(memory) ~  scale(sfc_all) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc_group))
summary(lmer(scale(memory) ~  scale(sfc_Hippocampus) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc_group))
summary(lmer(scale(memory) ~  scale(sfc_Subcortical) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc_group))
summary(lmer(scale(memory) ~  scale(sfc_Default) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc_group))
summary(lmer(scale(memory) ~  scale(sfc_Frontoparietal) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc_group))
summary(lmer(scale(memory) ~  scale(sfc_VentralAttention) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc_group))
summary(lmer(scale(memory) ~  scale(sfc_DorsalAttention) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc_group))

# Center variables
long_data_fc_group$memory_centered = scale(long_data_fc_group$memory)
long_data_fc_group$func_within_Subcortical_centered = scale(long_data_fc_group$func_within_Subcortical)

# 1) Fit the model
fc_group_mem <- lmer(memory_centered ~ func_within_Subcortical_centered * group + (1 | id) + sex + YoE , data = long_data_fc_group)
summary(fc_group_mem)

func_within_Subcortical_centered_seq <- seq(
  from = min(long_data_fc_group$func_within_Subcortical_centered, na.rm = TRUE),
  to   = max(long_data_fc_group$func_within_Subcortical_centered, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each superager_factor × sfc_all_centered
em_2group <- emmeans(
  fc_group_mem, 
  specs = ~ factor(group) * func_within_Subcortical_centered,
  at    = list(func_within_Subcortical_centered = func_within_Subcortical_centered_seq),
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
  mutate(func_within_Subcortical_centered = as.numeric(func_within_Subcortical_centered))  # Convert from factor if needed

# Define colors for each group
palette_1 <- c("maintainer" = "#60B5FF", "superager" = "pink")

# Create the two-group plot
ggplot() +
  geom_line(
    data = long_data_fc_group, 
    aes(x = func_within_Subcortical_centered, y = memory_centered, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data_fc_group, 
    aes(x = func_within_Subcortical_centered, y = memory_centered), 
    color = "gray", size = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = func_within_Subcortical_centered, ymin = lower_bound, ymax = upper_bound, 
        fill = factor(group)), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = func_within_Subcortical_centered, y = predicted, color = factor(group)), 
    size = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "func_within_Subcortical_centered", y = "memory_centered", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))

library(interactions)

# Fit model
fc_group_mem <- lmer(memory_centered ~ func_all_Default_centered * group + (1 | id) + sex + YoE, data = long_data_fc_group)

# Analyze the slopes for each group
sim_slopes(fc_group_mem, 
           pred = "func_all_Default_centered", 
           modx = "group",
           plot = FALSE)

# Fit model
sfc_da <- lmer(sfc_DorsalAttention_centered ~ age_centered * group + (1 | id) + sex + YoE , data = long_data_sfc_group)

# Analyze the slopes for each group
sim_slopes(sfc_da, 
           pred = "age_centered", 
           modx = "group",
           plot = FALSE)

struct_fpn_maint <- lmer(struct_within_Frontoparietal_centered ~ age_centered * maintainer_chr + (1 | id) + sex + YoE , data = long_data_struct)

sim_slopes(struct_fpn_maint, 
           pred = "age_centered", 
           modx = "maintainer_chr",
           plot = FALSE)

func_dmn_sa <- lmer(func_within_Default_centered ~ age_centered * superager_chr + (1 | id) + sex + YoE, data = long_data_fc)

sim_slopes(func_dmn_sa, 
           pred = "age_centered", 
           modx = "superager_chr",
           plot = FALSE)

func_sub <- lmer(scale(memory) ~  scale(func_within_Subcortical) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_fc_group)
sim_slopes(func_sub, 
           pred = "scale(func_within_Subcortical)", 
           modx = "group",
           plot = FALSE)

func_dmn <- lmer(scale(memory) ~  scale(func_within_Default) * group + (1 | id) + age + cohort + sex + YoE, data = long_data_fc_group)
sim_slopes(func_dmn, 
           pred = "scale(func_within_Default)", 
           modx = "group",
           plot = FALSE)


#######SYS 
summary(lmer(scale(memory) ~  scale(sys_v1) * superager_chr + (1 | id) + age + cohort + sex + YoE, data = long_data_sys))
summary(lmer(scale(memory) ~  scale(sys_v2) * superager_chr + (1 | id) + age + cohort + sex + YoE, data = long_data_sys))

summary(lmer(scale(memory) ~  scale(sys_v1) * maintainer_chr + (1 | id) + age + cohort + sex + YoE, data = long_data_sys))
summary(lmer(scale(memory) ~  scale(sys_v2) * maintainer_chr + (1 | id) + age + cohort + sex + YoE, data = long_data_sys))

summary(lmer(scale(memory) ~  scale(sys_v1) + (1 | id) + age + cohort + sex + YoE, data = long_data_sys))
summary(lmer(scale(memory) ~  scale(sys_v2) + (1 | id) + age + cohort + sex + YoE, data = long_data_sys))

summary(lmer(scale(sys_v1) ~  scale(age) * superager_chr + (1 | id) + cohort + sex + YoE, data = long_data_sys))
summary(lmer(scale(sys_v2) ~  scale(age) * superager_chr + (1 | id) + cohort + sex + YoE, data = long_data_sys))

summary(lmer(scale(sys_v1) ~  scale(age) * maintainer_chr + (1 | id) + cohort + sex + YoE, data = long_data_sys))
summary(lmer(scale(sys_v2) ~  scale(age) * maintainer_chr + (1 | id) + cohort + sex + YoE, data = long_data_sys))

summary(lmer(scale(sys_v1) ~  superager_chr + (1 | id) + age + cohort + sex + YoE, data = long_data_sys))
summary(lmer(scale(sys_v2) ~  superager_chr + (1 | id) + age + cohort + sex + YoE, data = long_data_sys))

summary(lmer(scale(sys_v1) ~  maintainer_chr + (1 | id) + age + cohort + sex + YoE, data = long_data_sys))
summary(lmer(scale(sys_v2) ~  maintainer_chr + (1 | id) + age + cohort + sex + YoE, data = long_data_sys))

# Center variables
long_data_sys$memory_centered = scale(long_data_sys$memory)
long_data_sys$sys_v1_centered = scale(long_data_sys$sys_v1)

# 1) Fit the model
sys <- lmer(memory_centered ~  sys_v1_centered * maintainer_chr + (1 | id) + age + cohort + sex + YoE, data = long_data_sys)
summary(sys)

sys_seq <- seq(
  from = min(long_data_sys$sys_v1_centered, na.rm = TRUE),
  to   = max(long_data_sys$sys_v1_centered, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each superager_factor × sfc_all_centered
em_2group <- emmeans(
  sys, 
  specs = ~ factor(maintainer_chr) * sys_v1_centered,
  at    = list(sys_v1_centered = sys_seq),
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
  mutate(sys_v1_centered = as.numeric(sys_v1_centered))  # Convert from factor if needed

# Define colors for each group
palette_1 <- c("maintainer" = "#60B5FF", "decliner" = "pink")

# Create the two-group plot
ggplot() +
  geom_line(
    data = long_data_sys, 
    aes(x = sys_v1_centered, y = memory_centered, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data_sys, 
    aes(x = sys_v1_centered, y = memory_centered), 
    color = "gray", size = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = sys_v1_centered, ymin = lower_bound, ymax = upper_bound, 
        fill = factor(maintainer_chr)), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = sys_v1_centered, y = predicted, color = factor(maintainer_chr)), 
    size = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "sys_v1_centered", y = "memory_centered", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))
