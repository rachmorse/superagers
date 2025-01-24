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

# Read in data
data <- read.csv("~/Documents/2023:2024/Data/Exported data/maintainer_superager_data.csv")

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

# Create a cohort variable 
data$cohort <- ifelse(data$id > 10000, "bbhi", "bbhi_senior")

# Reshape the data to long
long_data <- data %>%
  pivot_longer(
    cols = c(w1_age, w2_age, w1_memory, w2_memory),
    names_to = c("timepoint", ".value"),
    names_pattern = "w(\\d)_(.*)"
  )

##############################################
##### LME model - age, memory, 4 groups ######
##############################################

# Model with covariates (the model with the fig has no covariates)
model <- lmer(memory ~ age * superager_maintainer + (1 | id) + cohort + sex + YoE, data = long_data)
summary(model)

# Change the reference group to non-superager decliner (compare all groups to them)
long_data$superager_maintainer <- as.factor(long_data$superager_maintainer)
long_data$superager_maintainer <- relevel(long_data$superager_maintainer, ref = "non-superager decliner")

# Refit the model
model <- lmer(memory ~ age * superager_maintainer + (1 | id) + cohort + sex + YoE, data = long_data)
summary(model)

##############################################
##### LME model - age, memory, 2 groups ######
##############################################

# Model with covariates (the model with the fig has no covariates)
model_superager <- lmer(memory ~ age * superager + (1 | id) + cohort + sex + YoE, data = long_data)
summary(model_superager)

#############################################
##### Baseline memory four group model ######
#############################################

# Run linear regression
sa_mem_bl <- lm(w1_memory ~ superager_maintainer + w1_age + cohort + sex + YoE, data = data)
summary(sa_mem_bl)

# Change the reference group to non-superager decliner (compare all groups to them)
data$superager_maintainer <- as.factor(data$superager_maintainer)
data$superager_maintainer <- relevel(data$superager_maintainer, ref = "superager decliner")

##############################################
##### Follow-up memory four group model ######
##############################################

sa_mem_fu <- lm(w2_memory ~ superager_maintainer + w2_age + cohort + sex + YoE, data = data)
summary(sa_mem_fu)

#############################################
##### Baseline memory two group models ######
#############################################

# Run linear regression
superager_mem_bl <- aov(w1_memory ~ superager + w1_age + cohort + sex + YoE, data = data)
summary(superager_mem_bl)

maintainer_mem_bl <- aov(w1_memory ~ maintainer + w1_age + cohort + sex + YoE, data = data)
summary(maintainer_mem_bl)

##############################################
##### Follow-up memory two group models ######
##############################################

superager_mem_fu <- aov(w2_memory ~ superager + w2_age + cohort + sex + YoE, data = data)
summary(superager_mem_fu)

maintainer_mem_fu <- aov(w2_memory ~ maintainer + w2_age + cohort + sex + YoE, data = data)
summary(maintainer_mem_fu)

#####################################
## Fig 1: Two groups - age, memory ##
#####################################

model_res_lme <- lmer(memory ~ age * superager_factor + (1 | id), data = long_data)
summary(model_res_lme)

all_levels <- levels(factor(long_data$superager_factor))

# Generate predictions for each of the two group
predictions <- do.call(rbind, lapply(unique(long_data$superager_factor), function(res) {
  new_data <- data.frame(
    age = age_seq,
    superager_factor = factor(res, levels = all_levels),  # Ensure factor levels are consistent
    id = 1  # Assuming id doesn't impact fixed effects
  )
  
  # Obtain predicted values
  predicted <- predict(model_res_lme, newdata = new_data, re.form = NA)
  
  # Constructing design matrix to calculate standard errors
  X <- model.matrix(~ age * superager_factor, data = new_data)
  beta <- fixef(model_res_lme)
  Vb <- vcov(model_res_lme)
  se <- sqrt(diag(X %*% Vb %*% t(X)))
  
  new_data$predicted <- predicted
  new_data$lower_bound <- predicted - 1.96 * se
  new_data$upper_bound <- predicted + 1.96 * se
  new_data
}))

# Define colors for each group
palette_1 <- c("non-superager" = "#A35C7A", "superager" = "#FFD65A")

# Create the two group plot
fig1 <- ggplot() +
  geom_line(data = long_data, aes(x = age, y = memory, group = id), color = "lightgray", alpha = 0.5) + # Plot individual trajectories
  geom_point(data = long_data, aes(x = age, y = memory), color = "gray", size = 1) + # Add scatter points
  geom_ribbon(data = predictions, aes(x = age, ymin = lower_bound, ymax = upper_bound, fill = as.factor(superager_factor)), alpha = 0.3) + # Add CIs
  geom_line(data = predictions, aes(x = age, y = predicted, color = as.factor(superager_factor)), size = 1.2) + # Plot predicted lines for each group
  scale_color_manual(values = palette_1) +
  scale_fill_manual(values = palette_1) + # Ensure fill colors match line colors
  labs(x = "Age", y = "Episodic Memory Composite", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position = c(0.8, 0.94))

# Create figure captions 

# Divide the df into the two groups
superager_df <- long_data %>%
  filter(superager_factor == "superager")

non_superager_df <- long_data %>%
  filter(superager_factor == "non-superager")

# Run a scaled lmer for each group
superager_scaled <- lmer(scale(memory) ~ scale(age) + (1 | id), data = superager_df)
summary(superager_scaled)

non_superager_scaled <- lmer(scale(memory) ~ scale(age) + (1 | id), data = non_superager_df)
summary(non_superager_scaled)

######################################
## Fig 2: Four groups - age, memory ##
######################################

model_res_lme_maint <- lmer(memory ~ age * superager_maintainer + (1 | id), data = long_data)
summary(model_res_lme_maint)

# Generate predictions for each of the four group
predictions_2 <- do.call(rbind, lapply(unique(long_data$superager_maintainer), function(res) {
  new_data <- data.frame(
    age = age_seq,
    superager_maintainer = res,
    id = 1  
  )
  
  # Obtain predicted values
  predicted <- predict(model_res_lme_maint, newdata = new_data, re.form = NA)
  
  # Constructing design matrix to calculate standard errors
  X <- model.matrix(~ age * superager_maintainer, data = new_data)
  beta <- fixef(model_res_lme_maint)
  Vb <- vcov(model_res_lme_maint)
  se <- sqrt(diag(X %*% Vb %*% t(X)))
  
  new_data$predicted <- predicted
  new_data$lower_bound <- predicted - 1.96 * se
  new_data$upper_bound <- predicted + 1.96 * se
  new_data
}))

# Define colors for each group
palette <- c(
  "superager maintainer" = "#1f77b4",
  "superager decliner" = "#ff7f0e",
  "non-superager maintainer" = "#2ca02c",
  "non-superager decliner" = "#d62728"
)

# Create four group plot
fig2 <- ggplot() +
  geom_line(data = long_data, aes(x = age, y = memory, group = id), color = "lightgray", alpha = 0.5) + # Plot individual trajectories
  geom_point(data = long_data, aes(x = age, y = memory), color = "gray", size = 1) + # Add scatter points
  geom_ribbon(data = predictions_2, aes(x = age, ymin = lower_bound, ymax = upper_bound, fill = as.factor(superager_maintainer)), alpha = 0.3) + # Add CIs
  geom_line(data = predictions_2, aes(x = age, y = predicted, color = as.factor(superager_maintainer)), size = 1.2) + # Plot predicted lines for each group
  scale_color_manual(values = palette) +
  scale_fill_manual(values = palette) + # Ensure fill colors match line colors
  labs(x = "Age", y = "Episodic Memory Composite", color = "Group", fill = "Group") +
  theme_minimal() +
  theme(legend.position = c(0.72, 0.895))

# Combine Fig 1 and Fig 2 and save
combined_plot <- plot_grid(fig1, fig2, ncol = 2)
ggsave("SA vs nonSA.jpeg", plot = combined_plot, width = 10, height = 6, dpi = 400, units = "in")

# Create figure captions 

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
superager_maintainer_scaled <- lmer(scale(memory) ~ scale(age) + (1 | id), data = superager_maintainer_df)
summary(superager_maintainer_scaled)

non_superager_maintainer_scaled <- lmer(scale(memory) ~ scale(age) + (1 | id), data = non_superager_maintainer_df)
summary(non_superager_maintainer_scaled)

superager_decliner_scaled <- lmer(scale(memory) ~ scale(age) + (1 | id), data = superager_decliner_df)
summary(superager_decliner_scaled)

non_superager_decliner_scaled <- lmer(scale(memory) ~ scale(age) + (1 | id), data = non_superager_decliner_df)
summary(non_superager_decliner_scaled)
