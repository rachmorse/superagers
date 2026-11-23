# List participants with neuropsych comments to manually read

library(dplyr)
library(readxl)

# Paths
data_dir <- "~/Documents/2023:2024/Data"
export_dir <- file.path(data_dir, "Exported data")

# Comments that don't need reading
no_comment <- c("", " --", "A", "V")
no_comment_senior_tp1 <- c("", "NA", "ok", "yes")

# Read and prepare the data
ids <- read.csv(file.path(export_dir, "clean_data_all.csv"))
bbhi_raw_tp1 <- read.csv(file.path(data_dir, "BBHI", "BBHI Data Timept1 NPS.csv"))
bbhi_raw_tp2 <- read.csv(file.path(data_dir, "BBHI", "BBHI Data Timept2 NPS.csv"))
bbhi_senior_tp1 <- read_excel(file.path(data_dir, "BBHI-Senior", "bbhi senior data.xlsx"))
bbhi_senior_tp2 <- read.csv(file.path(data_dir, "BBHI-Senior", "Ministerio2024Wave2_DATA_2026-01-20_1245.csv"))

# Mutate and select for merge
ids <- ids %>%
  mutate(id = as.numeric(gsub("sub-", "", id))) %>%
  select(id)

bbhi_raw_tp1 <- bbhi_raw_tp1 %>%
  select(id, w1_nps_comment)

bbhi_raw_tp2 <- bbhi_raw_tp2 %>%
  select(id, nps_comment)

# Merge dfs
merged_data <- ids %>%
  left_join(bbhi_raw_tp1, by = "id") %>%
  left_join(bbhi_raw_tp2, by = "id") %>%
  arrange(id)

# Filter comments to only include those that need to be read
merged_data <- merged_data %>%
  filter(
    !(
      (is.na(w1_nps_comment) | w1_nps_comment %in% no_comment) &
        (is.na(nps_comment) | nps_comment %in% no_comment)
    )
  )

write.csv(merged_data, file.path(export_dir, "bbhi_nps_comments.csv"), row.names = FALSE)

# Repeat for BBHI senior
bbhi_senior_tp1 <- bbhi_senior_tp1 %>%
  rename(id = "ID") %>%
  select(id, NP_elegible_comments)

bbhi_senior_tp2 <- bbhi_senior_tp2 %>%
  rename(id = "record_id_np_w2") %>%
  select(id, comments_np_w2)

merged_bbhi_senior <- ids %>%
  left_join(bbhi_senior_tp1, by = "id") %>%
  left_join(bbhi_senior_tp2, by = "id") %>%
  arrange(id)

# "^sÃ" is "sí" with the broken encoding from the Excel file
merged_bbhi_senior <- merged_bbhi_senior %>%
  filter(
    !(
      (is.na(NP_elegible_comments) | NP_elegible_comments %in% no_comment_senior_tp1 |
        grepl("^sÃ", NP_elegible_comments)) &
        (is.na(comments_np_w2) | comments_np_w2 %in% no_comment)
    )
  )

write.csv(merged_bbhi_senior, file.path(export_dir, "bbhi_senior_nps_comments.csv"), row.names = FALSE)
