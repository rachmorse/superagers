if (!require("ggplot2")) {
  install.packages("ggplot2")
  require("ggplot2")
}
if (!require("dplyr")) {
  install.packages("dplyr")
  require("dplyr")
}
if (!require("tidyr")) {
  install.packages("tidyr")
  require("tidyr")
}
if (!require("stringr")) {
  install.packages("stringr")
  require("stringr")
}

hc_data_tp1 <- read.csv("~/Documents/2023:2024/Data/Exported data/hippocampus_and_wm_hypointensities_ses-01.csv")
hc_data_tp2 <- read.csv("~/Documents/2023:2024/Data/Exported data/hippocampus_and_wm_hypointensities_ses-02.csv")
age_data <- read.csv("~/Documents/2023:2024/Data/Exported data/maintainer_superager_data.csv")

# Create the timepoint variable
hc_data_tp1 <- hc_data_tp1 %>%
  mutate(timepoint = ifelse(grepl("ses-01", Measure.volume), 1, 2))

hc_data_tp2 <- hc_data_tp2 %>%
  mutate(timepoint = ifelse(grepl("ses-01", Measure.volume), 1, 2))

# Remove the session information from the id column
hc_data_tp1 <- hc_data_tp1 %>%
  mutate(id = str_replace(Measure.volume, "_ses-0[12]", ""))

hc_data_tp2 <- hc_data_tp2 %>%
  mutate(id = str_replace(Measure.volume, "_ses-0[12]", ""))

# Merge data
hc_data <- merge(hc_data_tp1, hc_data_tp2, by = intersect(names(hc_data_tp1), names(hc_data_tp2)), all = TRUE)

# Rename variables
hc_data <- hc_data %>%
  rename(
    wmh = "WM.hypointensities",
    icv = "EstimatedTotalIntraCranialVol",
    gm = "TotalGrayVol"
  ) 

# Calculate hippocampal total volume
hc_data <- hc_data %>% 
  mutate(hc = Left.Hippocampus + Right.Hippocampus)

# Select only need vars
hc_data <- hc_data %>% select(id, wmh, icv, hc, timepoint, gm)

# Drop subs with only one timepoint 
hc_data <- hc_data %>%
  group_by(id) %>%
  filter(n_distinct(timepoint) > 1) %>%
  ungroup()

# Reformat to wide 
wide_format <- function(data, n_timepoints) {
  result <- NULL
  
  for (i in 1:n_timepoints) {
    suffix <- paste0(".", i)
    timepoint_data <- data %>%
      filter(timepoint == i) %>%
      rename_with(~ paste0(.x, suffix), -c(id, timepoint)) %>%
      select(-starts_with("timepoint")) # Remove the timepoint column after processing
    
    result <- if (is.null(result)) {
      timepoint_data
    } else {
      full_join(result, timepoint_data, by = c("id"))
    }
  }
  
  return(result)
}

hc_data <- wide_format(hc_data, 2)

# Start by considering ICV for the HC measures

# The formula looks like: 
  
# Adjusted (HPC) volume = hc.1 (per participant) - b.1 (from the whole cohort) * (icv.1 (per participant) - mean_icv (from the whole cohort))
# Adjusted (HPC) volume = hc.2 - b.2 * (icv.1 (because it does not change from tp1) - mean_icv)

# Step 1. Calculate Mean ICV by timepoint 
mean_icv_by_timepoint <- hc_data %>%
  summarise(across(starts_with("icv"), mean, na.rm = TRUE))

# Rename the columns
mean_icv_by_timepoint <- mean_icv_by_timepoint %>%
  rename_with(~ gsub("icv", "icv_mean", .), starts_with("icv"))

# Add mean ICV values to hc_data as new columns
hc_data <- hc_data %>%
  mutate(icv_mean.1 = mean_icv_by_timepoint$icv_mean.1,
         icv_mean.2 = mean_icv_by_timepoint$icv_mean.2)

# Step 2. Define a function to calculate the regression slope b for each study at each timepoint
calculate_slope <- function(df, hip_total_cols, icv_cols) {
  slopes <- data.frame() # Create an empty dataframe to store the slopes
  
  for (i in seq_along(hip_total_cols)) {
    data <- df[, c(hip_total_cols[i], icv_cols[i])]
    
    # Skip if there is no data for this timepoint
    if (nrow(data) == 0 || all(is.na(data[[hip_total_cols[i]]]), is.na(data[[icv_cols[i]]]))) {
      next
    }
    
    model <- lm(data[[hip_total_cols[i]]] ~ data[[icv_cols[i]]], data = data, na.action = na.exclude)
    slope <- coef(model)[2] # Calculate the slope
    
    slopes <- rbind(slopes, data.frame(timepoint = i, slope = slope)) # Add the slope to the dataframe
  }
  
  return(slopes)
}

# Specify the columns to use for HIP_total (hc volume) and icv
hip_total_cols <- paste0("hc.", 1:2)
icv_cols <- paste0("icv.", 1:2)

# Calculate the slope for each study
slopes <- calculate_slope(hc_data, hip_total_cols, icv_cols)

# Convert the df to wide format to be able to merge it
slopes_wide <- slopes %>%
  pivot_wider(
    names_from = timepoint, values_from = c(-timepoint),
    names_prefix = "slope.", names_sep = "_"
  )

# Merge the wide slope df into the main df
hc_data <- hc_data %>%
  mutate(slope.1 = slopes_wide$slope.1,
         slope.2 = slopes_wide$slope.2)

# Step 3. Iterate a formula over each timepoint to calculate adjusted hippocampal volumes for each participant
for (timepoint in 1:2) {
  raw_volume_col <- paste0("hc.", timepoint)
  icv_col <- paste0("icv.", timepoint)
  icv_mean_col <- paste0("icv_mean.", timepoint)
  slope_col <- paste0("slope.", timepoint)
  adjusted_volume_col <- paste0("adj_hc.", timepoint)
  
  hc_data <- hc_data %>%
    mutate(!!adjusted_volume_col := .data[[raw_volume_col]] - .data[[slope_col]] * (.data[[icv_col]] - .data[[icv_mean_col]]))
}

# Check the output data to see if it makes sense
correlation <- cor(hc_data$hc.1, hc_data$adj_hc.1, use = "complete.obs")
print(correlation)

ggplot(hc_data, aes(x = hc.1, y = adj_hc.1)) +
  geom_point()

# Calculate hippocampus slope 

# First merge in ages
age_data <- age_data %>% 
  select(id, w1_age, w2_age, w1_memory, w2_memory)

# Add the string 'sub-' to the front of the ids
age_data <- age_data %>%
  mutate(id = paste0("sub-", id)) %>% 
  rename(
    age.1 = "w1_age",
    age.2 = "w2_age"
  )

hc_data <- merge(hc_data, age_data, by = "id")

calculate_annual_change <- function(data, n_timepoints) {
  # Calculate slopes
  for (i in 1:nrow(data)) {
    hc_data <- data[i, paste0("adj_hc.", 1:n_timepoints)]
    wmh_data <- data[i, paste0("wmh.", 1:n_timepoints)]
    age <- data[i, paste0("age.", 1:n_timepoints)]

    if (sum(!is.na(hc_data)) > 1) {
      data[i, "hc_slopes"] <- lm(hc_data[!is.na(hc_data)] ~ age[!is.na(hc_data)])$coefficients[2]
      data[i, "wmh_slopes"] <- lm(wmh_data[!is.na(wmh_data)] ~ age[!is.na(wmh_data)])$coefficients[2]
      data[i, "hc_time"] <- max(age[!is.na(hc_data)]) - min(age[!is.na(hc_data)])
    }
  }
  data
}

hc_data <- calculate_annual_change(hc_data, 2)

# Convert df to long format
long_data <- hc_data %>%
  pivot_longer(
    cols = matches("^age\\.\\d$|^adj_hc\\.\\d$|^wmh\\.\\d$"),
    names_to = c(".value", "timepoint"),
    names_pattern = "(.*)(\\d$)"
  )

long_data <- long_data %>%
  rename(
    age = "age.",
    adj_hc = "adj_hc.",
    wmh = "wmh."
  )

long_data <- long_data %>%
  filter(!is.na(age) & !is.na(adj_hc))

# Create the plot age v HC
ggplot(long_data, aes(x = age, y = adj_hc, group = id)) +
  geom_line(linewidth = 0.4, color = "lightblue") +
  geom_point(size = 0.6, color = "lightblue") +
  geom_smooth(method = "gam", formula = y ~ s(x, k = 24, bs = "cs"), aes(group = 1), color = "grey", se = TRUE) +
  labs(x = "Age", y = "Adjusted Hippocampal Volume") +
  theme_minimal() +
  theme(legend.position = "none") # Remove legend for the second plot

# Remove outliers to better see the data
long_data <- long_data %>%
  filter(wmh < 10000)

# Create the plot age v HC
ggplot(long_data, aes(x = age, y = wmh, group = id)) +
  geom_line(linewidth = 0.4, color = "blue") +
  geom_point(size = 0.6, color = "blue") +
  geom_smooth(method = "gam", formula = y ~ s(x, k = 24, bs = "cs"), aes(group = 1), color = "grey", se = TRUE) +
  labs(x = "Age", y = "White Matter Hypointensities") +
  theme_minimal() +
  theme(legend.position = "none") # Remove legend for the second plot

# Export data
hc_data <- hc_data %>% 
  select(id, paste0("wmh.", 1:2), paste0("adj_hc.", 1:2), paste0("gm.", 1:2),
         hc_slopes, wmh_slopes)

write.csv(hc_data, file = "~/Documents/2023:2024/Data/Exported data/hc_wml_wide.csv")
