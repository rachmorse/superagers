# Cleaning and prep ----

# Load necessary packages
library(dplyr)
library(tidyr)
library(stringr)
library(emmeans)
library(ggplot2)

# Read and prepare the data
data <- read.csv("~/Documents/2023:2024/Data/Exported data/clean_data_all.csv")

# Extract numeric id
data$id <- as.numeric(sub("sub-", "", data$id))

# Create cohort variable
data$cohort <- ifelse(data$id > 5000, "bbhi", "bbhi_senior")

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

long_data_fc_group <- long_data %>% 
  filter(!is.na(func_all_slopes) & !is.na(memory_slopes) & !is.na(group))

long_data_struct_group <- long_data %>% 
  filter(!is.na(struct_all_slopes) & !is.na(memory_slopes) & !is.na(group))

long_data_sfc_group <- long_data %>% 
  filter(!is.na(sfc_all_slopes) & !is.na(memory_slopes) & !is.na(group))

summary(lmer(scale(memory) ~  scale(func_all) + (1 | id) + age + cohort + sex + YoE, data = long_data_fc))
summary(lmer(scale(memory) ~  scale(func_within_DorsalAttention) + (1 | id)  + age + cohort + sex + YoE, data = long_data_fc))
summary(lmer(scale(memory) ~  scale(func_all_VentralAttention) + (1 | id)  + age + cohort + sex + YoE, data = long_data_fc))
summary(lmer(scale(memory) ~  scale(func_all_DorsalAttention) + (1 | id)  + age + cohort + sex + YoE, data = long_data_fc))
summary(lmer(scale(memory) ~  scale(func_Hippocampus) + (1 | id) + age + cohort + sex + YoE, data = long_data_fc))
summary(lmer(scale(memory) ~  scale(func_within_Subcortical) + (1 | id) + age + cohort + sex + YoE + cohort, data = long_data_fc))
summary(lmer(scale(memory) ~  scale(func_within_Default) + (1 | id) + age + cohort + sex + YoE + cohort, data = long_data_fc))
summary(lmer(scale(memory) ~  scale(func_within_Frontoparietal) + (1 | id) + age + cohort + sex + YoE, data = long_data_fc))
summary(lmer(scale(memory) ~  scale(func_within_VentralAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data_fc))
summary(lmer(scale(memory) ~  scale(func_all_Subcortical) + (1 | id) + age + cohort + sex + YoE, data = long_data_fc))
summary(lmer(scale(memory) ~  scale(func_all_Default) + (1 | id) + age + cohort + sex + YoE, data = long_data_fc))
summary(lmer(scale(memory) ~  scale(func_all_Frontoparietal) + (1 | id) + age + cohort + sex + YoE, data = long_data_fc))

summary(lmer(scale(memory) ~  scale(struct_all) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct))
summary(lmer(scale(memory) ~  scale(struct_Hippocampus) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct))
summary(lmer(scale(memory) ~  scale(struct_within_Subcortical) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct))
summary(lmer(scale(memory) ~  scale(struct_within_Default) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct))
summary(lmer(scale(memory) ~  scale(struct_within_Frontoparietal) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct))
summary(lmer(scale(memory) ~  scale(struct_within_VentralAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct))
summary(lmer(scale(memory) ~  scale(struct_within_DorsalAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct))
summary(lmer(scale(memory) ~  scale(struct_all_Subcortical) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct))
summary(lmer(scale(memory) ~  scale(struct_all_Default) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct))
summary(lmer(scale(memory) ~  scale(struct_all_Frontoparietal) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct))
summary(lmer(scale(memory) ~  scale(struct_all_VentralAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct))
summary(lmer(scale(memory) ~  scale(struct_all_DorsalAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data_struct))

summary(lmer(scale(memory) ~  scale(sfc_all) + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc))
summary(lmer(scale(memory) ~  scale(sfc_Hippocampus) + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc))
summary(lmer(scale(memory) ~  scale(sfc_Subcortical) + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc))
summary(lmer(scale(memory) ~  scale(sfc_Default) + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc))
summary(lmer(scale(memory) ~  scale(sfc_Frontoparietal) + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc))
summary(lmer(scale(memory) ~  scale(sfc_VentralAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc))
summary(lmer(scale(memory) ~  scale(sfc_DorsalAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data_sfc))

## THEN ##

summary(lmer(scale(func_all) ~  superager_chr + (1 | id) + sex + age + YoE, data = long_data_fc))
summary(lmer(scale(func_within_DorsalAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data_fc))
summary(lmer(scale(func_all_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data_fc))
summary(lmer(scale(func_all_DorsalAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data_fc))

summary(lmer(scale(func_all) ~  maintainer_chr + (1 | id) + sex + age + YoE, data = long_data_fc))
summary(lmer(scale(func_within_DorsalAttention) ~ maintainer_chr + (1 | id) + sex + age + YoE, data = long_data_fc))
summary(lmer(scale(func_all_VentralAttention) ~ maintainer_chr + (1 | id) + sex + age + YoE, data = long_data_fc))
summary(lmer(scale(func_all_DorsalAttention) ~ maintainer_chr + (1 | id) + sex + age + YoE, data = long_data_fc))

summary(lmer(scale(func_all) ~  group + (1 | id) + sex + age + YoE, data = long_data_fc_group))
summary(lmer(scale(func_within_DorsalAttention) ~ group + (1 | id) + sex + age + YoE, data = long_data_fc_group))
summary(lmer(scale(func_all_VentralAttention) ~ group + (1 | id) + sex + age + YoE, data = long_data_fc_group))
summary(lmer(scale(func_all_DorsalAttention) ~ group + (1 | id) + sex + age + YoE, data = long_data_fc_group))

summary(lmer(scale(sfc_all) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data_sfc))
summary(lmer(scale(sfc_Hippocampus) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data_sfc))
summary(lmer(scale(sfc_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data_sfc))

summary(lmer(scale(sfc_all) ~ maintainer_chr + (1 | id) + sex + age + YoE, data = long_data_sfc))
summary(lmer(scale(sfc_Hippocampus) ~ maintainer_chr + (1 | id) + sex + age + YoE, data = long_data_sfc))
summary(lmer(scale(sfc_VentralAttention) ~ maintainer_chr + (1 | id) + sex + age + YoE, data = long_data_sfc))

summary(lmer(scale(sfc_all) ~ group + (1 | id) + sex + age + YoE, data = long_data_sfc_group))
summary(lmer(scale(sfc_Hippocampus) ~ group + (1 | id) + sex + age + YoE, data = long_data_sfc_group))
summary(lmer(scale(sfc_VentralAttention) ~ group + (1 | id) + sex + age + YoE, data = long_data_sfc_group))

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
summary(lmer(scale(func_all) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_fc))
summary(lmer(scale(func_within_DorsalAttention) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_fc))
summary(lmer(scale(func_Hippocampus) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_fc))
summary(lmer(scale(func_within_Subcortical) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE + cohort, data = long_data_fc))
summary(lmer(scale(func_within_Default) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE + cohort, data = long_data_fc))
summary(lmer(scale(func_within_Frontoparietal) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_fc))
summary(lmer(scale(func_within_VentralAttention) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_fc))

summary(lmer(scale(func_all) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_fc))
summary(lmer(scale(func_within_DorsalAttention) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_fc))
summary(lmer(scale(func_Hippocampus) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_fc))
summary(lmer(scale(func_within_Subcortical) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_fc))
summary(lmer(scale(func_within_Default) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_fc))
summary(lmer(scale(func_within_Frontoparietal) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_fc))
summary(lmer(scale(func_within_VentralAttention) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_fc))

summary(lmer(scale(func_all) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_fc_group))
summary(lmer(scale(func_within_DorsalAttention) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_fc_group))
summary(lmer(scale(func_Hippocampus) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_fc_group))
summary(lmer(scale(func_within_Subcortical) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_fc_group))
summary(lmer(scale(func_within_Default) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_fc_group))
summary(lmer(scale(func_within_Frontoparietal) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_fc_group))
summary(lmer(scale(func_within_VentralAttention) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_fc_group))

# Apply FDR correction
pvals <- c(0.169612, 0.12668, 0.29080, 0.055381, 0.015479, 0.067109, 0.0466)
pvals_fdr <- p.adjust(pvals, method = "fdr")
print(pvals_fdr)

summary(lmer(scale(struct_all) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_struct))
summary(lmer(scale(struct_Hippocampus) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_struct))
summary(lmer(scale(struct_within_Subcortical) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_struct))
summary(lmer(scale(struct_within_Default) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_struct))
summary(lmer(scale(struct_within_Frontoparietal) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_struct))
summary(lmer(scale(struct_within_VentralAttention) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_struct))
summary(lmer(scale(struct_within_DorsalAttention) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_struct))

# Apply FDR correction
pvals <- c(0.002899, 0.01383, 0.00969, 0.011085, 0.006652, 0.04545, 0.02737)
pvals_fdr <- p.adjust(pvals, method = "fdr")
print(pvals_fdr)

summary(lmer(scale(struct_all) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_struct))
summary(lmer(scale(struct_Hippocampus) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_struct))
summary(lmer(scale(struct_within_Subcortical) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_struct))
summary(lmer(scale(struct_within_Default) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_struct))
summary(lmer(scale(struct_within_Frontoparietal) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_struct))
summary(lmer(scale(struct_within_VentralAttention) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_struct))
summary(lmer(scale(struct_within_DorsalAttention) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_struct))

summary(lmer(scale(struct_all) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(struct_Hippocampus) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(struct_within_Subcortical) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(struct_within_Default) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(struct_within_Frontoparietal) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(struct_within_VentralAttention) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_struct_group))
summary(lmer(scale(struct_within_DorsalAttention) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_struct_group))

summary(lmer(scale(sfc_all) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc))
summary(lmer(scale(sfc_Hippocampus) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc))
summary(lmer(scale(sfc_Subcortical) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc))
summary(lmer(scale(sfc_Default) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc))
summary(lmer(scale(sfc_Frontoparietal) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc))
summary(lmer(scale(sfc_VentralAttention) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc))
summary(lmer(scale(sfc_DorsalAttention) ~ maintainer_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc))

summary(lmer(scale(sfc_all) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc))
summary(lmer(scale(sfc_Hippocampus) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc))
summary(lmer(scale(sfc_Subcortical) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc))
summary(lmer(scale(sfc_Default) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc))
summary(lmer(scale(sfc_Frontoparietal) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc))
summary(lmer(scale(sfc_VentralAttention) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc))
summary(lmer(scale(sfc_DorsalAttention) ~ superager_chr * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc))

summary(lmer(scale(sfc_all) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc_group))
summary(lmer(scale(sfc_Hippocampus) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc_group))
summary(lmer(scale(sfc_Subcortical) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc_group))
summary(lmer(scale(sfc_Default) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc_group))
summary(lmer(scale(sfc_Frontoparietal) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc_group))
summary(lmer(scale(sfc_VentralAttention) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc_group))
summary(lmer(scale(sfc_DorsalAttention) ~ group * scale(age) + (1|id)   + sex + YoE, data = long_data_sfc_group))

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
long_data_fc_group$func_all_Default_centered = scale(long_data_fc_group$func_all_Default)

# 1) Fit the model
fc_group_mem <- lmer(memory_centered ~ func_all_Default_centered * group + (1 | id) + sex + YoE , data = long_data_fc_group)
summary(fc_group_mem)

func_all_Default_centered_seq <- seq(
  from = min(long_data_fc_group$func_all_Default_centered, na.rm = TRUE),
  to   = max(long_data_fc_group$func_all_Default_centered, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each superager_factor × sfc_all_centered
em_2group <- emmeans(
  fc_group_mem, 
  specs = ~ factor(group) * func_all_Default_centered,
  at    = list(func_all_Default_centered = func_all_Default_centered_seq),
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
  mutate(func_all_Default_centered = as.numeric(func_all_Default_centered))  # Convert from factor if needed

# Define colors for each group
palette_1 <- c("maintainer" = "#60B5FF", "superager" = "pink")

# Create the two-group plot
ggplot() +
  geom_line(
    data = long_data_fc_group, 
    aes(x = func_all_Default_centered, y = memory_centered, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data_fc_group, 
    aes(x = func_all_Default_centered, y = memory_centered), 
    color = "gray", size = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = func_all_Default_centered, ymin = lower_bound, ymax = upper_bound, 
        fill = factor(group)), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = func_all_Default_centered, y = predicted, color = factor(group)), 
    size = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "func_all_Default_centered", y = "memory_centered", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))

######################################
######### OPTION TWO #################
######################################
summary(lmer(scale(memory) ~  scale(func_all) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_DorsalAttention) + (1 | id)  + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_VentralAttention) + (1 | id)  + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_DorsalAttention) + (1 | id)  + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_VentralAttention) + (1 | id)  + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_DorsalAttention) + (1 | id)  + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_Hippocampus) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_Subcortical) + (1 | id) + age + cohort + sex + YoE + cohort, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_Default) + (1 | id) + age + cohort + sex + YoE + cohort, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_Frontoparietal) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_VentralAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_Subcortical) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_Default) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_Frontoparietal) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_Subcortical) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_Default) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_Frontoparietal) + (1 | id) + age + cohort + sex + YoE, data = long_data))

summary(lmer(scale(memory) ~  scale(struct_all) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_Hippocampus) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_Subcortical) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_Default) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_Frontoparietal) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_VentralAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_DorsalAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_Subcortical) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_Default) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_Frontoparietal) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_VentralAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_DorsalAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_Subcortical) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_Default) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_Frontoparietal) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_VentralAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_DorsalAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data))

summary(lmer(scale(memory) ~  scale(sfc_VentralAttention) + (1 | id)   + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Hippocampus) + (1 | id)  + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_all) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Hippocampus) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Subcortical) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Default) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Frontoparietal) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_VentralAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_DorsalAttention) + (1 | id) + age + cohort + sex + YoE, data = long_data))

## THEN ##

summary(lmer(scale(func_all) ~  superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_Hippocampus) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_within_Subcortical) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_within_Default) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_within_Frontoparietal) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_within_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_within_DorsalAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_between_Subcortical) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_between_Default) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_between_Frontoparietal) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_between_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_between_DorsalAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_all_Subcortical) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_all_Default) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_all_Frontoparietal) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_all_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_all_DorsalAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))

summary(lmer(scale(struct_all) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_Hippocampus) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_within_Subcortical) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_within_Default) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_within_Frontoparietal) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_within_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_within_DorsalAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_between_Subcortical) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_between_Default) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_between_Frontoparietal) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_between_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_between_DorsalAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_all_Subcortical) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_all_Default) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_all_Frontoparietal) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_all_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_all_DorsalAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))

summary(lmer(scale(sfc_all) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(sfc_Hippocampus) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(sfc_Subcortical) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(sfc_Default) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(sfc_Frontoparietal) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(sfc_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(sfc_DorsalAttention) ~  superager_chr + (1 | id) + sex + age + YoE, data = long_data))

## THEN mediation analysis for those that are significant

############################
###### LME Memory #########
############################

summary(lmer(scale(memory) ~  scale(func_all) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(func_all) ~  scale(age) * maintainer_chr + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(func_all) ~  scale(age) * superager_chr + (1 | id) + age + sex + YoE, data = long_data))

summary(lmer(scale(memory) ~  scale(func_within_DorsalAttention) + (1 | id)  + age + sex + YoE, data = long_data))
summary(lmer(scale(func_within_DorsalAttention) ~  scale(age) * maintainer_chr + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(func_within_DorsalAttention) ~  scale(age) * superager_chr + (1 | id) + age + sex + YoE, data = long_data))

summary(lmer(scale(memory) ~  scale(func_all_VentralAttention) + (1 | id)  + age + sex + YoE, data = long_data))
summary(lmer(scale(func_all_VentralAttention) ~  scale(age) * maintainer_chr + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(func_all_VentralAttention) ~  scale(age) * superager_chr + (1 | id) + age + sex + YoE, data = long_data))

summary(lmer(scale(memory) ~  scale(func_all_DorsalAttention) + (1 | id)  + age + sex + YoE, data = long_data))
summary(lmer(scale(func_all_DorsalAttention) ~  scale(age) * maintainer_chr + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(func_all_DorsalAttention) ~  scale(age) * superager_chr + (1 | id) + age + sex + YoE, data = long_data))

summary(lmer(scale(memory) ~  scale(func_between_VentralAttention) + (1 | id)  + age + sex + YoE, data = long_data))
summary(lmer(scale(func_between_VentralAttention) ~  scale(age) * maintainer_chr + (1 | id) + age  + sex + YoE, data = long_data))
summary(lmer(scale(func_between_VentralAttention) ~  scale(age) * superager_chr + (1 | id) + age + sex + YoE, data = long_data))

summary(lmer(scale(memory) ~  scale(func_between_DorsalAttention) + (1 | id)  + age + sex + YoE, data = long_data))
summary(lmer(scale(func_between_DorsalAttention) ~  scale(age) * maintainer_chr + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(func_between_DorsalAttention) ~  scale(age) * superager_chr + (1 | id) + age + sex + YoE, data = long_data))

summary(lmer(scale(memory) ~  scale(func_Hippocampus) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_Subcortical) + (1 | id) + age + sex + YoE + cohort, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_Default) + (1 | id) + age + sex + YoE + cohort, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_Frontoparietal) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_VentralAttention) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_Subcortical) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_Default) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_Frontoparietal) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_Subcortical) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_Default) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_Frontoparietal) + (1 | id) + age + sex + YoE, data = long_data))

summary(lmer(scale(memory) ~  scale(struct_all) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_Hippocampus) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_Subcortical) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_Default) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_Frontoparietal) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_VentralAttention) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_DorsalAttention) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_Subcortical) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_Default) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_Frontoparietal) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_VentralAttention) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_DorsalAttention) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_Subcortical) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_Default) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_Frontoparietal) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_VentralAttention) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_DorsalAttention) + (1 | id) + age + sex + YoE, data = long_data))

summary(lmer(scale(memory) ~  scale(sfc_VentralAttention) + (1 | id)   + age + sex + YoE, data = long_data))
summary(lmer(scale(sfc_VentralAttention) ~  scale(age) * superager_chr + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(sfc_VentralAttention) ~  scale(age) * maintainer_chr + (1 | id) + age + sex + YoE, data = long_data))

summary(lmer(scale(memory) ~  scale(sfc_Hippocampus) + (1 | id)  + age + sex + YoE, data = long_data))
summary(lmer(scale(sfc_Hippocampus) ~  scale(age) * superager_chr + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(sfc_Hippocampus) ~  scale(age) * maintainer_chr + (1 | id) + age + sex + YoE, data = long_data))

summary(lmer(scale(memory) ~  scale(sfc_all) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Hippocampus) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Subcortical) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Default) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Frontoparietal) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_VentralAttention) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_DorsalAttention) + (1 | id) + age + sex + YoE, data = long_data))

#########################################
###### LME Memory Interactions #########
#########################################

summary(lmer(scale(memory) ~  scale(func_all) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_Hippocampus) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_Subcortical) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_Default) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_Frontoparietal) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_VentralAttention) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_DorsalAttention) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_Subcortical) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_Default) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_Frontoparietal) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_VentralAttention) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_DorsalAttention) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_Subcortical) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_Default) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_Frontoparietal) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_VentralAttention) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_DorsalAttention) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))

summary(lmer(scale(memory) ~  scale(struct_all) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_Hippocampus) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_Subcortical) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_Default) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_Frontoparietal) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_VentralAttention) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_DorsalAttention) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_Subcortical) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_Default) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_Frontoparietal) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_VentralAttention) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_DorsalAttention) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_Subcortical) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_Default) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_Frontoparietal) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_VentralAttention) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_DorsalAttention) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))

summary(lmer(scale(memory) ~  scale(sfc_all) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Hippocampus) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Subcortical) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Default) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Frontoparietal) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_VentralAttention) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_DorsalAttention) * scale(superager) + (1 | id) + age + sex + YoE, data = long_data))

# Plot significant interaction

# Center variables
long_data$sfc_all_centered = scale(long_data$sfc_all)
long_data$maintainer_centered = scale(long_data$maintainer)
long_data$memory_centered = scale(long_data$memory)

# 1) Fit the model
model_sfc_lme <- lmer(memory_centered ~ sfc_all_centered * maintainer_centered + (1 | id) + age+ sex + YoE, data = long_data)
summary(model_sfc_lme)

# 2) Create a grid of ages for which we want predictions
sfc_all_centered_seq <- seq(
  from = min(long_data$sfc_all_centered, na.rm = TRUE),
  to   = max(long_data$sfc_all_centered, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each superager_factor × sfc_all_centered
em_2group <- emmeans(
  model_sfc_lme, 
  specs = ~ factor(maintainer_centered) * sfc_all_centered,
  at    = list(sfc_all_centered = sfc_all_centered_seq),
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
  mutate(sfc_all_centered = as.numeric(sfc_all_centered))  # Convert from factor if needed

# Define colors for each group
palette_1 <- c("1.95553885959982" = "#A35C7A", "-0.509254911354119" = "#FFD65A")

# Create the two-group plot
ggplot() +
  geom_line(
    data = long_data, 
    aes(x = sfc_all_centered, y = memory_centered, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data, 
    aes(x = sfc_all_centered, y = memory_centered), 
    color = "gray", size = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = sfc_all_centered, ymin = lower_bound, ymax = upper_bound, 
        fill = factor(maintainer_centered)), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = sfc_all_centered, y = predicted, color = factor(maintainer_centered)), 
    size = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "SFC", y = "memory_centered", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))

#########################################
###### LME Interactions #########
#########################################

summary(lmer(scale(func_all) ~  superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_Hippocampus) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_within_Subcortical) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_within_Default) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_within_Frontoparietal) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_within_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_within_DorsalAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_between_Subcortical) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_between_Default) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_between_Frontoparietal) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_between_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_between_DorsalAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_all_Subcortical) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_all_Default) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_all_Frontoparietal) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_all_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(func_all_DorsalAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))

summary(lmer(scale(struct_all) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_Hippocampus) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_within_Subcortical) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_within_Default) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_within_Frontoparietal) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_within_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_within_DorsalAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_between_Subcortical) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_between_Default) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_between_Frontoparietal) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_between_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_between_DorsalAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_all_Subcortical) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_all_Default) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_all_Frontoparietal) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_all_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(struct_all_DorsalAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))

summary(lmer(scale(sfc_all) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(sfc_Hippocampus) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(sfc_Subcortical) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(sfc_Default) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(sfc_Frontoparietal) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(sfc_VentralAttention) ~ superager_chr + (1 | id) + sex + age + YoE, data = long_data))
summary(lmer(scale(sfc_DorsalAttention) ~  superager_chr + (1 | id) + sex + age + YoE, data = long_data))


#########################################
###### LME Memory Age Interactions #########
#########################################


summary(lmer(scale(memory) ~  scale(func_all) * superager_chr * scale(age) * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_Hippocampus) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_Subcortical) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_Default) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_Frontoparietal) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_VentralAttention) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_within_DorsalAttention) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_Subcortical) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_Default) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_Frontoparietal) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_VentralAttention) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_between_DorsalAttention) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_Subcortical) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_Default) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_Frontoparietal) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_VentralAttention) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(func_all_DorsalAttention) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))

summary(lmer(scale(memory) ~  scale(struct_all) * superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_Hippocampus) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_Subcortical) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_Default) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_Frontoparietal) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_VentralAttention) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_within_DorsalAttention) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_Subcortical) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_Default) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_Frontoparietal) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_VentralAttention) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_between_DorsalAttention) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_Subcortical) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_Default) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_Frontoparietal) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_VentralAttention) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(struct_all_DorsalAttention) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))

summary(lmer(scale(memory) ~  scale(sfc_all) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Hippocampus) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Subcortical) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Default) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_Frontoparietal) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_VentralAttention) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))
summary(lmer(scale(memory) ~  scale(sfc_DorsalAttention) *  superager_chr * scale(age) + (1 | id) + sex + YoE, data = long_data))

# 1) Fit the model
long_data$func_within_Default_centered <- scale(long_data$func_within_Default)
long_data$age_centered <- scale(long_data$age)

model_sfc_lme <- lmer(func_within_Default_centered ~ age_centered * superager_chr  + (1 | id) + sex + YoE, data = long_data)
summary(model_sfc_lme)

long_data_superager <- long_data %>% 
  filter(
    superager == 1
  )

long_data_nonsuperager <- long_data %>% 
  filter(
    superager == 0
  )
model_sfc_lme <- lmer(scale(memory) ~ scale(func_within_Default_centered)  + (1 | id) + age + sex + YoE, data = long_data_nonsuperager)
summary(model_sfc_lme)

# 2) Create a grid of ages for which we want predictions
age_centered_seq <- seq(
  from = min(long_data$age_centered, na.rm = TRUE),
  to   = max(long_data$age_centered, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each superager_factor × age_centered
em_2group <- emmeans(
  model_sfc_lme, 
  specs = ~ superager_chr * age_centered,
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
    data = long_data, 
    aes(x = age_centered, y = func_within_Default_centered, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data, 
    aes(x = age_centered, y = func_within_Default_centered), 
    color = "gray", size = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = age_centered, ymin = lower_bound, ymax = upper_bound, 
        fill = superager_chr), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = age_centered, y = predicted, color = superager_chr), 
    size = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "age", y = "func_within_Default_centered", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))

model_memory_func <- lmer(scale(memory) ~ scale(func_all) + (1 | id) + age + sex + YoE + cohort, data = long_data)
summary(model_memory_func)

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
