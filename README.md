## Superagers Research Study

This repository contains the analysis scripts for our study on superagers, which explores their longitudinal memory and structural and functional connectivity in aging. This document provides an overview of the scripts.

## Table of Contents

- [Overview](#overview)
- [Folders](#folders)

## Overview

This repository contains several folders with different steps in the analyis process including Classification, fMRI analysis and structural_analysis.

The folders, listed in their intended order of use, are described below along with an overview of the files included in the repository.

## Folders

1. **Classification**
    - **Purpose:** These scripts clean the data from the BBHI and BBHI senior cohorts to be able to merge the relevant data, then classify participants as superagers or non-superagers and as maintainers or decliners. 
    - **Scripts:**
        - `cleaning BBHI data.ipynb`: Cleans BBHI data for merge.
        - `cleaning BBHI senior data.ipynb`: Cleans BBHI senior data for merge.
        - `superager_classification.ipynb`: Classifies participants as superagers or non-superagers using [Sun et al. (2016)](https://pubmed.ncbi.nlm.nih.gov/27629716/) criteria:
            - Superagers must:
                - Score at or above the mean for age 16-29 year olds on the RAVLT long delay free recall based on normative data from [Schmidt (1996)](https://scholar.google.co.uk/scholar?hl=en&as_sdt=0%2C5&q=Schmidt%2C+M.+%281996%29.+Rey+Auditory+and+Verbal+Learning+Test%3A+A+handbook.+Los+Angeles%2C+CA%3A+Western+Psychological+Services&btnG=)
                - Score above 1 SD below the norm for age and education on the TMT B based on the neuronorma data 
                - Score above 1.5 SD below the norm for age and education on the TMT A and B, semantic fluency, digit span forward and backward based on the neuronorma data 
        - `maintainer_classification.ipynb`: Classifies participants as maintainers or decliners. 
            - Maintainers must: 
                - Have memory change that is equal to or greater than 0.
    - **Notes:**
        - This folder also includes Excel files with the published neuronorma data from [Peña-Casanova et al. (2009a)](https://pubmed.ncbi.nlm.nih.gov/19661109/) and [Peña-Casanova et al. (2009b)](https://pubmed.ncbi.nlm.nih.gov/19648583/) that were used to determine participants who were within 1 SD of the norm on the neuropsychological tests. 

1. **fMRI analysis**
    - **Purpose:** These scripts scrub the preprocessed fMRI data, extract timeseries data, compute functional connectivity correlations and conduct statistical significance testing on the correlations. 
    - **Scripts:**
        - `01_scrubbing_fMRI.py`: Scrubs fMRI BOLD images based on a Framewise Displacement (FWD) threshold to help mitigate motion artifacts. It either removes or interpolates frames where a subject has a high FWD (i.e. 0.5).
        - `02_extract_subjects_timeseries.py`: Extracts timeseries data from fMRI BOLD images using the Schaefer 200 ROI, 7 network atlas. 
            - Uses the functions in the script `extract_timeseries.py`.
        - `03_compute_subject_functional_connectivity.py`: Computes various functional connectivity metrics for subjects from the timeseries data. The script processes timeseries data, computes functional connectivity with and without Fisher z-transformation, and saves the results to CSV files.
            - Uses the functions in the script `compute_functional_connectivity.py`.
        - `04_compute_group_connectivity`: Computes the average functional connectivy for given groups (e.g., superagers or non-superagers) cross-sectionally.
        - `05_stats_test_thresholded.py`: Thresholds connectivity matrices and runs t-tests to compare connectivity by group. This thresholds the data by group (e.g. top 15% for superagers, top 15% for non-superagers and then keeps the *all* columns that survive thresholding from both groups)
        - `05_stats_test`: Runs t-tests to compare unthresholded connectivity by group. 
        - `05_threshold_all_subjects.py`: Thresholds connectivity matrices and runs t-tests to compare connectivity by group. This thresholds the data for the whole cohort (e.g. top 15% for the whole cohort) 
    - **Notes:**
        - The scripts used to preprocesses the fMRI data used here are available in another [repository](https://github.com/rachmorse/fmri_preprocessing)
