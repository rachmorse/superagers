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
- [Misc Scripts](#misc-scripts)

## Folders

### classification
- **Purpose:** These scripts clean the data from the BBHI and BBHI senior cohorts to be able to merge the relevant data, then classify participants as superagers or non-superagers. 
- **Scripts:**
    - `check_invalid_nps`: Prepares BBHI and BBHI senior raw data, filtering for those with comments about their neuropsychological data. It saves the comments, so they can be read manually to exclude subjects with invalid data. 
    - `cleaning_bbhi_data.ipynb`: Cleans BBHI data for merge.
    - `cleaning_bbhi_senior_data.ipynb`: Cleans BBHI senior data for merge.
    - `superager_classification.ipynb`: Classifies participants as superagers or non-superagers using modified [Sun et al. (2016)](https://pubmed.ncbi.nlm.nih.gov/27629716/) and [Rogalski et al. (2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3541673/)] criteria:
        - Superagers must:
            - Score at or above the mean for age 16-29 year olds on the RAVLT long delay free recall based on normative data from [Schmidt (1996)](https://scholar.google.co.uk/scholar?hl=en&as_sdt=0%2C5&q=Schmidt%2C+M.+%281996%29.+Rey+Auditory+and+Verbal+Learning+Test%3A+A+handbook.+Los+Angeles%2C+CA%3A+Western+Psychological+Services&btnG=) at baseline and follow-up. 
            - Score within 1 SD of the norm for age and education on the TMT B, inverse digits and semantic fluency based on the neuronorma data from [Peña-Casanova et al. (2009a)](https://pubmed.ncbi.nlm.nih.gov/19661109/) with Spanish adults at baseline and follow-up. 

### fsaverage
- **Purpose:** These scripts transform the fsaverage Schaefer atlas and aseg parcellation to T1 native space then DWI native space and BOLD native T1 space and combine the two atlases to have a Schaefer + subcortical atlas for each subject using Freesurfer and FSL. 
- **Scripts:**
    - `01_fsaverage_to_t1.py`: Transforms the Schaefer 200 ROI atlas from FreeSurfer’s ‘fsaverage’ surface space to each subject’s T1 space. It uses FreeSurfer's mri_surf2surf (converts to subject specific surface) and mri_label2vol (conducts surface-to-volume projection in native T1 space).
    - `02_subcortical_to_t1.py`: Extracts subcortical regions from FreeSurfer's aseg.mgz files for each subject, creating labeled volumetric masks in T1 space using mri_vol2vol (resamples from aseg to T1 space).
    - `03_combine_t1_atlases.py`: Stacks left and right Schaefer cortical and subcortical masks into a single subject-specific atlas in T1 space, handling overlaps to preserve all 214 brain regions.
    - `04_t1_to_dwi_bold.py`: Transforms the Schaefer/subcortical atlases from T1 space into native DWI and native-T1 BOLD space using FSL's flirt and epi_reg for DWI images and FreeSurfer's mri_vol2vol for BOLD images. 
    - `run_pipeline.py`: Runs all scripts above, looping through each timepoint and cohort.  

### fmri_analysis
- **Purpose:** These scripts scrub the preprocessed fMRI data, extract timeseries data, and compute functional connectivity correlations. 
- **Scripts:**
    - `01_scrubbing_fMRI.py`: Scrubs fMRI BOLD images based on a Framewise Displacement (FWD) threshold to help mitigate motion artifacts. It interpolates frames where a subject has a high FWD.
    - `02_extract_subjects_timeseries.py`: Extracts timeseries data from fMRI BOLD images. NOTE - this script excludes participants who had >30% of their frames scrubbed.
        - Uses the functions in the script `extract_timeseries.py`.
    - `03_compute_subject_functional_connectivity.py`: Computes various functional connectivity metrics for subjects from the timeseries data. The script processes timeseries data, computes functional connectivity with and without Fisher z-transformation, and saves the results to CSV files.
        - Uses the functions in the script `compute_functional_connectivity.py`.
- **Notes:**
    - The scripts used to preprocesses the fMRI data used here are available in another [repository](https://github.com/rachmorse/fmri_preprocessing).

### structural_analysis
- **Purpose:** These scripts use Multi-Shell Multi-Tissue Constrained Spherical Deconvolution (MSMT-CSD) to calculate white matter tracts using FSL, SPM, FreeSurfer and MRTrix and extract the structural connectivity matrices data using MRTrix. 
- **Scripts:**
    - `tractography_parallelized.py`: Runs the complete tractography pipeline. It handles parallel processing of subjects to perform rigid-body coregistration (structural to diffusion), tissue response function estimation, Fibre Orientation Distributions (FODs) estimation using MSMT-CSD, and Anatomically-Constrained Tractography (ACT) with SIFT2 to add weighting to the final tractograms.
        - Uses `spm_coregister_parcellation.m`: A MATLAB script used for the rigid-body coregistration step. It aligns the structural T1 and parcellation files to the diffusion b0 image using Normalized Mutual Information (NMI), modifying the NIfTI headers in-place.
    - `generate_structural_matrices.py`: Computes various structural connectivity matrices from the MRTrix data include all-to-all ROI and network specific matrices. Also, optionally visualizes the matrices. 
        - Uses functions from the script `compute_functional_connectivity.py`.
- **Notes:**
    - The scripts used in DWI preprocessing are available in another [repository](https://github.com/rachmorse/dwi_preprocessing). 

### structure_function_coupling
- **Purpose:** These scripts run prepare and calculate the structural functional coupling (SFC) metrics. To calculate SFC, we use the Pearson's correlation coefficient between the row for a given ROI in the structural connectome and the corresponding row in the functional connectome. We exclude self-connections and any connections where either the structural or functional connectivity value equaled zero (method from this [paper](https://doi.org/10.1038/s41467-023-41686-9)).
- **Scripts:**
    - `01_convert_to_individual_matrix.py`: Converts the functional and structural connectivity data from a shared CSV with all participants into individual 214x214 matrices. 
    - `02_structural_functional_coupling.py`: Computes and visualizes SFC using the normalized structural connectivity data and untransformed functional connectivity data. 
    - `03_group_means.py`: Consolidates individual SFC data into a single group matrix, applies Fisher z-transform, calculates and saves group means, and creates visualizations.
   
### elastic_net
- **Purpose:** These scripts run prepare and run a logistic elastic net to classify participants as superagers or non-superagers
- **Scripts:**
    `prep_data_for_en.py`: Prepares data by generating voxel‑weighted, grouped ROI connectivity summaries for later elastic net analysis. 
    `prep_weighted_global_roi_averages.py`: Computes voxel-weighted global, sensory, and heteromodal ROI averages for SFC, FC, and SC.
    `log_en.py`: Runs a logistic elastic net classification to predict superager status using cross-validation and permutation testing. Provides model level metrics including p-value and feature level p-values. 

## Analysis Scripts

1. **`draft_analysis_2.R`**
    - **Purpose:** The first exploritory analysis.  
2. **`elastic_net.py`**
    - **Purpose:** Run an elastic net analysis to extract ROIs that are most important in memory. 

## Misc Scripts
1. **`missing_subs.py`**
    - **Purpose:** Calculates which subs have functional data but to not have structural data and viceversa to make sure no subjects are missing data unecessarily.
