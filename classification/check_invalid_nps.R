library(dplyr)
library(tidyr)
library(readxl)

# Read and prepare the data
ids <- read.csv("~/Documents/2023:2024/Data/Exported data/clean_data_all.csv")
bbhi_raw_tp2 <- read.csv("~/Documents/2023:2024/Data/BBHI/BBHI Data Timept2 NPS.csv")
bbhi_raw_tp1 <- read.csv("~/Documents/2023:2024/Data/BBHI/BBHI Data Timept1 NPS.csv")
bbhi_senior_tp1 <- read_excel("~/Documents/2023:2024/Data/BBHI-Senior/bbhi senior data.xlsx")
bbhi_senior_tp2 <- read.csv("~/Documents/2023:2024/Data/BBHI-Senior/Ministerio2024Wave2_DATA_2025-05-14_1043.csv")

# Mutate and select for merge
ids <- ids %>% 
  mutate(id = as.numeric(gsub("sub-", "", id))) %>% 
  select(id)

bbhi_raw_tp1 <- bbhi_raw_tp1 %>% 
  select(id, w1_nps_comment)

bbhi_raw_tp2 <- bbhi_raw_tp2 %>% 
  select(id, nps_comment)

# Merge dfs
merged_df <- merge(ids, bbhi_raw_tp1, by = "id", all.x = TRUE)
merged_data <- merge(merged_df, bbhi_raw_tp2, by = "id", all.x = TRUE)

# Filter comments to only include those that need to be read
merged_data <- merged_data %>%
  filter(
    !(
      (is.na(w1_nps_comment) | w1_nps_comment == "" | w1_nps_comment == " --" | w1_nps_comment == "A" | w1_nps_comment == "V") &
        (is.na(nps_comment)   | nps_comment == ""   | nps_comment == " --" | nps_comment == "A" | nps_comment == "V")
    )
  )

write.csv(merged_data, "~/Downloads/bbhi_nps_comments.csv", row.names = FALSE)

# Repeat for BBHI senior
bbhi_senior_tp1 <- bbhi_senior_tp1 %>% 
  rename(id = "ID") %>% 
  select(id, NP_elegible_comments)

bbhi_senior_tp2 <- bbhi_senior_tp2 %>% 
  rename(id = "record_id_np_w2") %>% 
  select(id, comments_np_w2)

merged_bbhi_senior <- merge(ids, bbhi_senior_tp1, by = "id", all.x = TRUE)
merged_bbhi_senior <- merge(merged_bbhi_senior, bbhi_senior_tp2, by = "id", all.x = TRUE)

merged_bbhi_senior <- merged_bbhi_senior %>%
  filter(
    !(
      (is.na(NP_elegible_comments) | NP_elegible_comments == "" | 
         NP_elegible_comments == "NA" | NP_elegible_comments == "ok" | 
         NP_elegible_comments == "yes" | grepl("^sÃ", NP_elegible_comments)) 
    )
  )

write.csv(merged_bbhi_senior, "~/Downloads/bbhi_senior_nps_comments.csv", row.names = FALSE)
