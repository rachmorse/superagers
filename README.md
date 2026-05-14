## Superagers Research Study

This repository contains the analysis scripts from our study on superagers, which explores their structural connectivity, functional connectivity, and structure-function coupling. 

## Usage
This repository provides details on the analyses for transparency. The dataset used is not included, though it can be requested with appropriate ethical approval.

## License
The code is available under the MIT License, allowing others to reuse and adapt it with appropriate credit.

## Table of Contents

- [Folders](#folders)
  - [superager_classification](#superager_classification)
  - [fsaverage_masks](#fsaverage)
  - [fmri_analysis](#fmri-analysis)
  - [structural_analysis](#structural_analysis)
  - [structure_function_coupling](#structure_function_coupling)
  - [elastic_net](#elastic_net)
  - [analyses](#analyses)

## Folders

### superager_classification
- **Purpose:** Clean data from our cohort to merge the relevant files and classify participants as superagers or non-superagers. 
- **Scripts:**
    - `check_invalid_nps.R`: Saves comments about participants' neuropsychological data, to be read manually and remove invalid data. 
    - `cleaning_bbhi_data.ipynb`: Cleans BBHI data for merge.
    - `cleaning_bbhi_senior_data.ipynb`: Cleans BBHI senior data for merge.
    - `superager_classification.ipynb`: Classifies participants as superagers or non-superagers. 

### fsaverage_masks
- **Purpose:** Transform the fsaverage Schaefer-200 atlas and recon-all aseg parcellation to combined atlases in native DWI and native-T1 BOLD space.
- **Scripts:**
    - `01_fsaverage_to_t1.py`: Transforms Schaefer atlas from fsaverage to each subject’s T1 space. 
    - `02_subcortical_to_t1.py`: Extracts subcortical regions from aseg.mgz files.
    - `03_combine_t1_atlases.py`: Stacks Schaefer cortical and subcortical masks into a single subject-specific atlas in T1 space.
    - `04_t1_to_dwi_bold.py`: Transforms the Schaefer/subcortical atlases from T1 space into native DWI and native-T1 BOLD space. 
    - `run_pipeline.py`: Runs all scripts above, looping through each timepoint and cohort.  

### fmri_analysis
- **Purpose:** Scrub the preprocessed fMRI data, extract timeseries data, and compute functional connectivity correlations. 
- **Scripts:**
    - `01_scrubbing_fMRI.py`: Scrubs BOLD images based on a Framewise Displacement (FWD) threshold. 
    - `02_extract_subjects_timeseries.py`: Extracts timeseries data from BOLD images using Schaefer/subcortical atlases. 
        - Uses functions in `extract_timeseries.py`.
    - `03_compute_subject_functional_connectivity.py`: Processes timeseries data and computes Fisher z-transformation.
        - Uses functions in `compute_functional_connectivity.py`.
- **Notes:**
    - The scripts used to preprocesses fMRI data are available in another [repository](https://github.com/rachmorse/fmri_preprocessing).

### structural_analysis
- **Purpose:** Uses Multi-Shell Multi-Tissue Constrained Spherical Deconvolution (MSMT-CSD) to calculate white matter tracts and extract structural connectivity matrices. 
- **Scripts:**
    - `tractography_parallelized.py`: Runs complete tractography pipeline. Performs rigid-body coregistration (structural to diffusion), tissue response function estimation, Fibre Orientation Distribution (FOD) estimation using MSMT-CSD, and Anatomically-Constrained Tractography (ACT) with SIFT2.
        - Uses `spm_coregister_parcellation.m`: For the rigid-body coregistration. 
    - `generate_structural_matrices.py`: Computes structural connectivity matrices. 
        - Uses functions in `compute_functional_connectivity.py`.
- **Notes:**
    - The scripts used in DWI preprocessing are available in another [repository](https://github.com/rachmorse/dwi_preprocessing). 

### structure_function_coupling
- **Purpose:** Prepare and calculate structure-function coupling (SFC) metrics. 
- **Scripts:**
    - `01_convert_to_individual_matrix.py`: Converts functional and structural connectivity data into individual 214x214 matrices. 
    - `02_structural_functional_coupling.py`: Computes and visualizes SFC. 
    - `plot_group_connectivity_figure.py`: Plots a single subject's structural connectivity matrix, functional connectivity matrix, and SFC vector.

### elastic_net
- **Purpose:** Prepare and run logistic elastic net to classify participants as superagers or non-superagers.
- **Scripts:**
    - `prep_data_for_en.py`: Prepares data as per ROI summaries. 
    - `prep_weighted_global_roi_averages.py`: Computes voxel-weighted global, sensory, heteromodal, and memory-relevant network averages for SFC.
    - `log_en.py`: Runs logistic elastic net classification predicting superager status using cross-validation and permutation testing. 
    - `en_fdr.py`: Reads elastic net nohup log files, extracts model-level p-values, and reports FDR-adjusted p-values for each model.
    - `plot_feature_importance.py`: Plots the top 20 ROIs from SFC elastic net models as bar charts. Includes a brain plot panel with cortical and subcortical regions. 
    - `make_supplementary_table.py`: Generates supplementary table listing top 50 features from each SFC elastic net model.

### analyses

- **Purpose:** Conduct all non-elastic net related analyses.
- **Scripts:**  
    - `analyses.R`: Runs full results pipeline: mixed-effects models for episodic memory, superager status, and SFC with FDR correction.
    - `results.Rmd`: R Markdown document that sources `analyses.R` and auto-fills the results section with model outputs.
    - `plot_sfc_difference.py`: Averages superager vs. non-superager SFC difference maps and saves a cortical/subcortical figure.
    - `plot_sfc_em_cortex.py`: Combines the figure from `plot_sfc_difference.py` with forest plots of stats from `results.html`.
