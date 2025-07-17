if (!require("dplyr")) {
  install.packages("dplyr")
  require("dplyr")
}
if (!require("lme4")) {
  install.packages("lme4")
  require("lme4")
}
if (!require("lmerTest")) {
  install.packages("lmerTest")
  require("lmerTest")
}
if (!require("tidyr")) {
  install.packages("tidyr")
  require("tidyr")
}
if (!require("ggplot2")) {
  install.packages("ggplot2")
  require("ggplot2")
}
if (!require("merTools")) {
  install.packages("merTools")
  require("merTools")
}
if (!require("cowplot")) {
  install.packages("cowplot")
  require("cowplot")
}
if (!require("emmeans")) {
  install.packages("emmeans")
  require("emmeans")
}

# Read in data
data <- read.csv("~/Documents/2023:2024/Data/Exported data/maintainer_superager_data.csv")
hc_data <- read.csv("~/Documents/2023:2024/Data/Exported data/hc_wml_wide.csv")

# Remove "sub-" prefix from id and convert to numeric
hc_data <- hc_data %>%
  mutate(id = as.numeric(gsub("sub-", "", id)))

data <- merge(hc_data, data, by = "id")

####################################
## Create variables / format data ##
####################################

# Create the four groups
data <- data %>%
  mutate(superager_maintainer = case_when(
    maintainer == 1 & superager == 1 ~ "superager maintainer",
    maintainer == 0 & superager == 1 ~ "superager decliner",
    maintainer == 1 & superager == 0 ~ "non-superager maintainer",
    maintainer == 0 & superager == 0 ~ "non-superager decliner",
    TRUE ~ "unknown"
  ))

# Create the maintainer groups
data <- data %>%
  mutate(maintainer_factor = case_when(
    maintainer == 1 ~ "maintainer",
    maintainer == 0 ~ "decliner",
    TRUE ~ "unknown"
  ))

# Create a cohort variable 
data$cohort <- ifelse(data$id > 10000, "bbhi", "bbhi_senior")

# Reshape the data to long
names(data) <- sub("^w(\\d)_(.*)", "\\2.\\1", names(data))

long_data <- data %>%
  pivot_longer(
    cols = matches("^age\\.\\d$|^wmh\\.\\d$|^memory\\.\\d$|^adj_hc\\.\\d$|^gm\\.\\d$"),
    names_to = c(".value", "timepoint"),
    names_pattern = "(.*)(\\d$)"
  )

long_data <- long_data %>%
  rename(
    age = "age.",
    memory = "memory.",
    wmh = "wmh.",
    adj_hc = "adj_hc.",
    gm = "gm."
  )

##############################################
##### LME model - age, memory, 4 groups ######
##############################################
# Convert superager_maintainer to a factor
long_data <- long_data %>%
  mutate(superager_maintainer = factor(superager_maintainer))

# Model with covariates (the model with the fig has no covariates)
model <- lmer(memory ~ age * superager_maintainer + (1 | id) + cohort + sex + YoE, data = long_data)
summary(model)

# Change the reference group to non-superager decliner (compare all groups to them)
long_data$superager_maintainer <- as.factor(long_data$superager_maintainer)
long_data$superager_maintainer <- relevel(long_data$superager_maintainer, ref = "non-superager decliner")

# Refit the model
model <- lmer(memory ~ age * superager_maintainer + (1 | id) + cohort  + sex + YoE, data = long_data)
summary(model)

##############################################
##### LME model - age, memory, 2 groups ######
##############################################
# SUPERAGERS 

# Model with covariates (the model with the fig has no covariates)
model_superager <- lmer(memory ~ age * superager + (1 | id) + cohort  + sex + YoE, data = long_data)
summary(model_superager)

##############################################
##### LME model - age, memory, 2 groups ######
##############################################
# MAINTAINERS 

# Model with covariates (the model with the fig has no covariates)
model_maintainer <- lmer(memory ~ age * maintainer + (1 | id) + cohort  + sex + YoE, data = long_data)
summary(model_maintainer)

#############################################
##### Baseline memory four group model ######
#############################################

# Run linear regression
sa_mem_bl <- lm(memory.1 ~ superager_maintainer + age.1  + sex + YoE, data = data)
summary(sa_mem_bl)

# Change the reference group to non-superager decliner (compare all groups to them)
data$superager_maintainer <- as.factor(data$superager_maintainer)
data$superager_maintainer <- relevel(data$superager_maintainer, ref = "non-superager maintainer")

##############################################
##### Follow-up memory four group model ######
##############################################

sa_mem_fu <- lm(memory.2 ~ superager_maintainer + age.2  + sex + YoE, data = data)
summary(sa_mem_fu)

#############################################
##### Baseline memory two group models ######
#############################################

# Run linear regression
superager_mem_bl <- aov(memory.1 ~ superager + age.1  + sex + YoE, data = data)
summary(superager_mem_bl)

maintainer_mem_bl <- aov(memory.1 ~ maintainer + age.1  + sex + YoE, data = data)
summary(maintainer_mem_bl)

##############################################
##### Follow-up memory two group models ######
##############################################

superager_mem_fu <- aov(memory.2 ~ superager + age.2  + sex + YoE, data = data)
summary(superager_mem_fu)

maintainer_mem_fu <- aov(memory.2 ~ maintainer + age.2  + sex + YoE, data = data)
summary(maintainer_mem_fu)

#####################################
## Fig 1: Two groups - age, memory ##
#####################################

###### CURRENTLY USING EMMEANS BUT CONSIDER SWITCHING TO GGEFFECTS

long_data <- long_data %>%
  mutate(superager_factor = case_when(
    superager == 1 ~ "superager",
    superager == 0 ~ "non-superager",
    TRUE ~ "unknown"
  ))

# 1) Fit the model
model_res_lme <- lmer(memory ~ age * superager_factor + (1 | id) + cohort  + sex + YoE, data = long_data)
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
  specs = ~ superager_factor * age,
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
palette_1 <- c("non-superager" = "#A35C7A", "superager" = "#FFD65A")

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
        fill = superager_factor), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = age, y = predicted, color = superager_factor), 
    size = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "Age", y = "Episodic Memory Composite", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))
fig1

# 5) Subset data for each group to run separate scaled models
superager_df <- long_data %>% filter(superager_factor == "superager")
non_superager_df <- long_data %>% filter(superager_factor == "non-superager")

superager_scaled <- lmer(scale(memory) ~ scale(age) + (1 | id) + cohort  + sex + YoE, data = superager_df)
summary(superager_scaled)

non_superager_scaled <- lmer(scale(memory) ~ scale(age) + (1 | id) + cohort  + sex + YoE, data = non_superager_df)
summary(non_superager_scaled)

######################################
## Fig 2: Four groups - age, memory ##
######################################

model_res_lme_maint <- lmer(memory ~ age * superager_maintainer + (1 | id) + cohort  + sex + YoE, data = long_data)
summary(model_res_lme_maint)

# 6) Similarly, use emmeans for the 4-group model
em_4group <- emmeans(
  model_res_lme_maint, 
  specs = ~ superager_maintainer * age,
  at    = list(age = age_seq),
  reff  = 0
)

predictions_2 <- summary(em_4group) %>%
  as.data.frame() %>%
  rename(
    predicted   = emmean,
    lower_bound = lower.CL,
    upper_bound = upper.CL
  ) %>%
  mutate(age = as.numeric(age))

# Define colors for each group
palette <- c(
  "superager maintainer"     = "#1f77b4",
  "superager decliner"       = "#ff7f0e",
  "non-superager maintainer" = "#2ca02c",
  "non-superager decliner"   = "#d62728"
)

# 7) Create 4-group plot
fig2 <- ggplot() +
  geom_line(
    data = long_data, 
    aes(x = age, y = memory, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data, 
    aes(x = age, y = memory), 
    color = "gray", linewidth = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions_2, 
    aes(x = age, ymin = lower_bound, ymax = upper_bound, 
        fill = superager_maintainer), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions_2, 
    aes(x = age, y = predicted, color = superager_maintainer), 
    linewidth = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette) +
  scale_fill_manual(values = palette) +
  labs(x = "Age", y = "Episodic Memory Composite", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.72, 0.895))

# 8) Combine Fig 1 and Fig 2 and save
combined_plot <- plot_grid(fig1, fig2, ncol = 2)
ggsave("SA vs nonSA.jpeg", plot = combined_plot, width = 10, height = 6, dpi = 400, units = "in")
combined_plot

# Divide the df into the four groups
superager_maintainer_df <- long_data %>%
  filter(superager_maintainer == "superager maintainer")

non_superager_maintainer_df <- long_data %>%
  filter(superager_maintainer == "non-superager maintainer")

superager_decliner_df <- long_data %>%
  filter(superager_maintainer == "superager decliner")

non_superager_decliner_df <- long_data %>%
  filter(superager_maintainer == "non-superager decliner")

# Run a scaled lmer for each group
superager_maintainer_scaled <- lmer(scale(memory) ~ scale(age) + (1 | id) + cohort  + sex + YoE, data = superager_maintainer_df)
summary(superager_maintainer_scaled)

non_superager_maintainer_scaled <- lmer(scale(memory) ~ scale(age) + (1 | id) + cohort  + sex + YoE, data = non_superager_maintainer_df)
summary(non_superager_maintainer_scaled)

superager_decliner_scaled <- lmer(scale(memory) ~ scale(age) + (1 | id) + cohort  + sex + YoE, data = superager_decliner_df)
summary(superager_decliner_scaled)

non_superager_decliner_scaled <- lmer(scale(memory) ~ scale(age) + (1 | id) + cohort  + sex + YoE, data = non_superager_decliner_df)
summary(non_superager_decliner_scaled)

##########################################
## Fig 3: Two groups - age, hippocampus ##
##########################################

# 1) Fit the model
model_hc_lme <- lmer(adj_hc ~ age * superager_factor + (1 | id) + cohort  + sex + YoE, data = long_data)
summary(model_hc_lme)

# 2) Create a grid of ages for which we want predictions
age_seq <- seq(
  from = min(long_data$age, na.rm = TRUE),
  to   = max(long_data$age, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each superager_factor × age
em_2group <- emmeans(
  model_hc_lme, 
  specs = ~ superager_factor * age,
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
palette_1 <- c("non-superager" = "#A35C7A", "superager" = "#FFD65A")

# Create the two-group plot
ggplot() +
  geom_line(
    data = long_data, 
    aes(x = age, y = adj_hc, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data, 
    aes(x = age, y = adj_hc), 
    color = "gray", linewidth = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = age, ymin = lower_bound, ymax = upper_bound, 
        fill = superager_factor), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = age, y = predicted, color = superager_factor), 
    linewidth = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "Age", y = "Adjusted Hippocampal Volume", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))

##########################################
## Fig 4: Two groups - age, hippocampus ##
##########################################

# 1) Fit the model
model_hc_lme <- lmer(adj_hc ~ age * maintainer_factor + (1 | id) + cohort  + sex + YoE, data = long_data)
summary(model_hc_lme)

# 2) Create a grid of ages for which we want predictions
age_seq <- seq(
  from = min(long_data$age, na.rm = TRUE),
  to   = max(long_data$age, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each superager_factor × age
em_2group <- emmeans(
  model_hc_lme, 
  specs = ~ maintainer_factor * age,
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
palette_1 <- c("maintainer" = "#A35C7A", "decliner" = "#FFD65A")

# Create the two-group plot
ggplot() +
  geom_line(
    data = long_data, 
    aes(x = age, y = adj_hc, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data, 
    aes(x = age, y = adj_hc), 
    color = "gray", linewidth = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = age, ymin = lower_bound, ymax = upper_bound, 
        fill = maintainer_factor), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = age, y = predicted, color = maintainer_factor), 
    linewidth = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "Age", y = "Adjusted Hippocampal Volume", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))

##################################
## Fig 5: Two groups - age, wmh ##
##################################
# Remove outliers to better see the data
long_data <- long_data %>%
  filter(wmh < 10000)

# 1) Fit the model
model_wmh_lme <- lmer(wmh ~ age * superager_factor + (1 | id) + cohort  + sex + YoE, data = long_data)
summary(model_wmh_lme)

# 2) Create a grid of ages for which we want predictions
age_seq <- seq(
  from = min(long_data$age, na.rm = TRUE),
  to   = max(long_data$age, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each superager_factor × age
em_2group <- emmeans(
  model_wmh_lme, 
  specs = ~ superager_factor * age,
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
palette_1 <- c("non-superager" = "#A35C7A", "superager" = "#FFD65A")

# Create the two-group plot
ggplot() +
  geom_line(
    data = long_data, 
    aes(x = age, y = wmh, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data, 
    aes(x = age, y = wmh), 
    color = "gray", linewidth = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = age, ymin = lower_bound, ymax = upper_bound, 
        fill = superager_factor), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = age, y = predicted, color = superager_factor), 
    linewidth = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "Age", y = "White Matter Hypointensities", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))

##################################
## Fig 6: Two groups - age, whm ##
##################################

# 1) Fit the model
model_wmh_lme <- lmer(wmh ~ age * maintainer_factor + (1 | id) + cohort  + sex + YoE, data = long_data)
summary(model_wmh_lme)

# 2) Create a grid of ages for which we want predictions
age_seq <- seq(
  from = min(long_data$age, na.rm = TRUE),
  to   = max(long_data$age, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each superager_factor × age
em_2group <- emmeans(
  model_wmh_lme, 
  specs = ~ maintainer_factor * age,
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
palette_1 <- c("maintainer" = "#A35C7A", "decliner" = "#FFD65A")

# Create the two-group plot
ggplot() +
  geom_line(
    data = long_data, 
    aes(x = age, y = wmh, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data, 
    aes(x = age, y = wmh), 
    color = "gray", linewidth = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = age, ymin = lower_bound, ymax = upper_bound, 
        fill = maintainer_factor), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = age, y = predicted, color = maintainer_factor), 
    linewidth = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "Age", y = "White Matter Hypointensities", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))

###################################
## Fig 7: Four groups - age, wmh ##
###################################

model_res_lme_maint <- lmer(wmh ~ age * superager_maintainer + (1 | id) + cohort  + sex + YoE, data = long_data)
summary(model_res_lme_maint)

# 6) Similarly, use emmeans for the 4-group model
em_4group <- emmeans(
  model_res_lme_maint, 
  specs = ~ superager_maintainer * age,
  at    = list(age = age_seq),
  reff  = 0
)

predictions_2 <- summary(em_4group) %>%
  as.data.frame() %>%
  rename(
    predicted   = emmean,
    lower_bound = lower.CL,
    upper_bound = upper.CL
  ) %>%
  mutate(age = as.numeric(age))

# Define colors for each group
palette <- c(
  "superager maintainer"     = "#1f77b4",
  "superager decliner"       = "#ff7f0e",
  "non-superager maintainer" = "#2ca02c",
  "non-superager decliner"   = "#d62728"
)

# 7) Create 4-group plot
ggplot() +
  geom_line(
    data = long_data, 
    aes(x = age, y = wmh, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data, 
    aes(x = age, y = wmh), 
    color = "gray", linewidth = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions_2, 
    aes(x = age, ymin = lower_bound, ymax = upper_bound, 
        fill = superager_maintainer), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions_2, 
    aes(x = age, y = predicted, color = superager_maintainer), 
    linewidth = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette) +
  scale_fill_manual(values = palette) +
  labs(x = "Age", y = "WMH", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.72, 0.895))

##################################
## Fig 8: Four groups - age, hc ##
##################################

model_res_lme_maint <- lmer(adj_hc ~ age * superager_maintainer + (1 | id) + cohort  + sex + YoE, data = long_data)
summary(model_res_lme_maint)

# 6) Similarly, use emmeans for the 4-group model
em_4group <- emmeans(
  model_res_lme_maint, 
  specs = ~ superager_maintainer * age,
  at    = list(age = age_seq),
  reff  = 0
)

predictions_2 <- summary(em_4group) %>%
  as.data.frame() %>%
  rename(
    predicted   = emmean,
    lower_bound = lower.CL,
    upper_bound = upper.CL
  ) %>%
  mutate(age = as.numeric(age))

# Define colors for each group
palette <- c(
  "superager maintainer"     = "#1f77b4",
  "superager decliner"       = "#ff7f0e",
  "non-superager maintainer" = "#2ca02c",
  "non-superager decliner"   = "#d62728"
)

# 7) Create 4-group plot
ggplot() +
  geom_line(
    data = long_data, 
    aes(x = age, y = adj_hc, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data, 
    aes(x = age, y = adj_hc), 
    color = "gray", linewidth = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions_2, 
    aes(x = age, ymin = lower_bound, ymax = upper_bound, 
        fill = superager_maintainer), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions_2, 
    aes(x = age, y = predicted, color = superager_maintainer), 
    linewidth = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette) +
  scale_fill_manual(values = palette) +
  labs(x = "Age", y = "Hippocampal Volume", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.72, 0.895))

#################################
## Fig 9: Two groups - age, gm ##
#################################

# 1) Fit the model
model_gm_lme <- lmer(gm ~ age * maintainer_factor + (1 | id) + cohort  + sex + YoE, data = long_data)
summary(model_gm_lme)

# 2) Create a grid of ages for which we want predictions
age_seq <- seq(
  from = min(long_data$age, na.rm = TRUE),
  to   = max(long_data$age, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each superager_factor × age
em_2group <- emmeans(
  model_gm_lme, 
  specs = ~ maintainer_factor * age,
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
palette_1 <- c("maintainer" = "#A35C7A", "decliner" = "#FFD65A")

# Create the two-group plot
ggplot() +
  geom_line(
    data = long_data, 
    aes(x = age, y = gm, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data, 
    aes(x = age, y = gm), 
    color = "gray", linewidth = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = age, ymin = lower_bound, ymax = upper_bound, 
        fill = maintainer_factor), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = age, y = predicted, color = maintainer_factor), 
    linewidth = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "Age", y = "Total Grey Matter", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))

###################################
## Fig 10: Four groups - age, gm ##
###################################

model_res_lme_maint <- lmer(gm ~ age * superager_maintainer + (1 | id) + cohort  + sex + YoE, data = long_data)
summary(model_res_lme_maint)

# 6) Similarly, use emmeans for the 4-group model
em_4group <- emmeans(
  model_res_lme_maint, 
  specs = ~ superager_maintainer * age,
  at    = list(age = age_seq),
  reff  = 0
)

predictions_2 <- summary(em_4group) %>%
  as.data.frame() %>%
  rename(
    predicted   = emmean,
    lower_bound = lower.CL,
    upper_bound = upper.CL
  ) %>%
  mutate(age = as.numeric(age))

# Define colors for each group
palette <- c(
  "superager maintainer"     = "#1f77b4",
  "superager decliner"       = "#ff7f0e",
  "non-superager maintainer" = "#2ca02c",
  "non-superager decliner"   = "#d62728"
)

# 7) Create 4-group plot
ggplot() +
  geom_line(
    data = long_data, 
    aes(x = age, y = gm, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data, 
    aes(x = age, y = gm), 
    color = "gray", linewidth = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions_2, 
    aes(x = age, ymin = lower_bound, ymax = upper_bound, 
        fill = superager_maintainer), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions_2, 
    aes(x = age, y = predicted, color = superager_maintainer), 
    linewidth = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette) +
  scale_fill_manual(values = palette) +
  labs(x = "Age", y = "Total Grey Matter", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.72, 0.895))

#################################
## Fig 11: Two groups - age, gm ##
#################################

# 1) Fit the model
model_gm_lme <- lmer(gm ~ age * superager_factor + (1 | id) + cohort  + sex + YoE, data = long_data)
summary(model_gm_lme)

# 2) Create a grid of ages for which we want predictions
age_seq <- seq(
  from = min(long_data$age, na.rm = TRUE),
  to   = max(long_data$age, na.rm = TRUE),
  length.out = 100
)

# 3) Use emmeans to get marginal predictions for each superager_factor × age
em_2group <- emmeans(
  model_gm_lme, 
  specs = ~ superager_factor * age,
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
palette_1 <- c("superager" = "#A35C7A", "non-superager" = "#FFD65A")

# Create the two-group plot
ggplot() +
  geom_line(
    data = long_data, 
    aes(x = age, y = gm, group = id), 
    color = "lightgray", alpha = 0.5
  ) +  # Plot individual trajectories
  geom_point(
    data = long_data, 
    aes(x = age, y = gm), 
    color = "gray", linewidth = 1
  ) +  # Add scatter points
  geom_ribbon(
    data = predictions, 
    aes(x = age, ymin = lower_bound, ymax = upper_bound, 
        fill = superager_factor), 
    alpha = 0.3
  ) +  # Add CIs
  geom_line(
    data = predictions, 
    aes(x = age, y = predicted, color = superager_factor), 
    linewidth = 1.2
  ) +  # Plot predicted lines
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) +  # Match line & fill colors
  labs(x = "Age", y = "Total Grey Matter", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position.inside = c(0.8, 0.94))

