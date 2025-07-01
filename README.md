## Superagers Research Study

This repository contains the analysis scripts for our study on superagers, which explores their longitudinal memory and structural and functional connectivity in aging. This document provides an overview of the scripts.

## Table of Contents

- [Folders](#folders)
  - [classification](#classification)
  - [fsaverage_masks](#fsaverage)
  - [fmri_analysis](#fmri-analysis)
  - [structural_analysis](#structural_analysis)
  - [structural_functional_coupling](#structural_functional_coupling)
- [Analysis Scripts](#analysis-scripts)

## Folders

### classification
- **Purpose:** These scripts clean the data from the BBHI and BBHI senior cohorts to be able to merge the relevant data, then classify participants as superagers or non-superagers and as maintainers or decliners. 
- **Scripts:**
    - `check_invalid_nps`: Prepares BBHI and BBHI senior raw data, filtering for those with comments about their neuropsychological data. It saves the comments, so they can be read manually to exclude subjects with invalid data. 
    - `cleaning_bbhi_data.ipynb`: Cleans BBHI data for merge.
    - `cleaning_bbhi_senior_data.ipynb`: Cleans BBHI senior data for merge.
    - `superager_classification.ipynb`: Classifies participants as superagers or non-superagers using modified [Sun et al. (2016)](https://pubmed.ncbi.nlm.nih.gov/27629716/) criteria:
        - All participants must:
            - Have an MMSE score >= 27 (I suggest that this allows us to consider them as healthy agers, rather than the stricter Sun et al. criteria)
        - Superagers must:
            - Score at or above the mean for age 16-29 year olds on the RAVLT long delay free recall based on normative data from [Schmidt (1996)](https://scholar.google.co.uk/scholar?hl=en&as_sdt=0%2C5&q=Schmidt%2C+M.+%281996%29.+Rey+Auditory+and+Verbal+Learning+Test%3A+A+handbook.+Los+Angeles%2C+CA%3A+Western+Psychological+Services&btnG=).
            - Score above 1 SD below the norm for age and education on the TMT B based on the neuronorma data from [Peña-Casanova et al. (2009a)](https://pubmed.ncbi.nlm.nih.gov/19661109/) with Spanish adults.
    - `maintainer_classification.ipynb`: Classifies participants as maintainers or decliners. 
        - Maintainers must: 
            - Have memory change and baseline memory that are greater than the average.

### fsaverage_masks
- **Purpose:** These scripts transform the fsaverage Schaefer atlas and aseg parcellation to T1 native space then DWI and BOLD native space and combine the two atlases to have native space Schaefer + subcortical atlas for each subject. 
- **Scripts:**
    - `01_fsaverage_to_t1.py`: Transforms the Schaefer 200-parcel atlas from FreeSurfer’s ‘fsaverage’ surface space to each subject’s T1 space. It uses FreeSurfer's mri_surf2surf and mri_label2vol.
    - `02_subcortical_to_t1.py`: Extracts subcortical regions from FreeSurfer's aseg.mgz files for each subject, creating labeled volumetric masks T1 space.
    - `03_combine_t1_atlases.py`: Merges left and right Schaefer cortical and subcortical masks into a single subject-specific atlas in T1 space, handling overlaps and preserving all 214 brain regions.
    - `04_t1_to_dwi_bold.py`: Transforms the Schaefer/subcortical atlases from T1 space into native DWI and BOLD spaces, generating native-space masks for downstream structural and functional analyses. It uses FSL's FLIRT. 

### fmri_analysis
- **Purpose:** These scripts scrub the preprocessed fMRI data, extract timeseries data, compute functional connectivity correlations and conduct statistical significance testing on the correlations. 
- **Scripts:**
    - `01_scrubbing_fMRI.py`: Scrubs fMRI BOLD images based on a Framewise Displacement (FWD) threshold to help mitigate motion artifacts. It either removes or interpolates frames where a subject has a high FWD (i.e. 0.5).
    - `02_extract_subjects_timeseries.py`: Extracts timeseries data from fMRI BOLD images using the Schaefer 200 ROI, 7 network atlas and 14 aseg-derived subcortical regions. NOTE - this script excludes participants who had >30% of their frames scrubbed.
        - Uses the functions in the script `extract_timeseries.py`.
    - `03_compute_subject_functional_connectivity.py`: Computes various functional connectivity metrics for subjects from the timeseries data. The script processes timeseries data, computes functional connectivity with and without Fisher z-transformation, and saves the results to CSV files.
        - Uses the functions in the script `compute_functional_connectivity.py`.
    - `04_compute_group_connectivity`: Computes the average functional connectivy for given groups (e.g., superagers or non-superagers) cross-sectionally.
    - `05_stats_test_thresholded.py`: Thresholds connectivity matrices and runs t-tests to compare connectivity by group. This thresholds the data by group (e.g. top 15% for superagers, top 15% for non-superagers and then keeps the *all* columns that survive thresholding from both groups)
    - `05_stats_test`: Runs t-tests to compare unthresholded connectivity by group. 
    - `05_threshold_all_subjects.py`: Thresholds connectivity matrices and runs t-tests to compare connectivity by group. This thresholds the data for the whole cohort (e.g. top 15% for the whole cohort) 
- **Notes:**
    - The scripts used to preprocesses the fMRI data used here are available in another [repository](https://github.com/rachmorse/fmri_preprocessing).

### structural_analysis
- **Purpose:** These scripts extract the structural data using FreeSurfer. 
- **Scripts:**
    - `extract_freesurfer_stats.py`: Extracts the structural data following the recon-all processing including hippocampal volumes and white matter hypointensities.
    - `generate_structural_matrices.py`: Computes various structural connectivity matrices from the MRTrix data include all-to-all ROI and network specific matrices. Also, visualizes the matrices. 
        - Uses functions from the script `compute_functional_connectivity.py`.
- **Notes:**
    - The script uses subprocess and a wrapper to run in Python 2 to be compatable with Freesurfer 6.

### structural_functional_coupling
- **Purpose:** These scripts run prepare and calculate the structural functional coupling (SFC) metrics. To calculate SFC, we use the Pearson correlation coefficient between the row for a given ROI in the structural connectome and the corresponding row in the functional connectome. We exclude self-connections and any connections where either the structural or functional connectivity values equal zero (method from this [paper](https://doi.org/10.1038/s41467-023-41686-9)).
 
- **Scripts:**
    - `01_convert_to_individual_matrix.py`: Converts the functional and structural connectivity data from a shared CSV with all participants into individual 214x214 matrices. 
        - NOTE this remove functional connectivtiy value <0, replacing them with 0. 
    - `02_structural_functional_coupling.py`: Computes and visualizes SFC. 
    - `03_group_means.py`: Consolidates individual SFC data into a single group matrix, applies Fisher z-transform, merges with superager/maintainer status, calculates and saves group means, and creates visualizations.
    - `04_calculate_network_metrics.py`: Calculates average within- and between-network functional and structural connectivity, and structure-function coupling for each of the 7 Schaefer networks.
    - `05_annual_change.py`: Calculates annual change slopes for all connectivity and SFC network metrics and merges these slopes with the memory and demographic data, creating a clean dataframe with all data for analysis. 

## Analysis Scripts

1. **`calculate_adj_hippocampus.R`**
    - **Purpose:** Calculates the adjusted hippocampal volume as well as white matter hypointensity and hippocampal slopes.
2. **`AAIC Superager Abstract.R`**
    - **Purpose:** Conducts analysis of the longitudinal memory trajectories and structural trajectories for all groups (e.g., non-superager decliners, superager maintainers)
3. **`analysis.ipynb`**
    - **Purpose:** The first exploritory analysis. Similar to `AAIC Superager Abstract.R` but a few steps behind and with some additional analyses not used for the AAIC abstract 
4. **`gephi_visualization_prep.py`**
    - **Purpose:** Reorganizes the connectivty matrices data to be able to be used for creating Gephi figures. 

## Misc Scripts
1. **`missing_subs.py`**
    - **Purpose:** Calculates which subs have functional data but to not have structural data and viceversa to make sure no subjects are missing data unecessarily.