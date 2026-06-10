#!/usr/bin/env python3
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_DIR = Path('/home/rachel/Desktop/data')
RAW_DIR = DATA_DIR / 'consort_raw'
SFC_DIR = Path('/home/rachel/Desktop/schaefer_analysis/structure_function_coupling')
SCHAEFER_DIR = Path('/home/rachel/Desktop/schaefer_analysis/fsaverage')
BBHI_MRI = Path('/pool/guttmann/institut/BBHI/MRI')
SENIOR_MRI = Path('/pool/guttmann/institut/UB/Superagers/MRI')
BBHI_RECONALL = BBHI_MRI / 'derivatives/reconall_fs6'
SENIOR_RECONALL = SENIOR_MRI / 'derivatives/reconall_fs6'
SCRUBBING_LOG = Path('/home/rachel/Desktop/superagers/fmri_analysis/nohup_timeseries.out')


def has_reconall(sub, ses):
    """Check whether FreeSurfer recon-all output exists for a subject/session.

    Args:
        sub (str): Subject ID in the format 'sub-xxx'.
        ses (int): Session number (1 or 2).

    Returns:
        bool: True if the recon-all directory exists, False otherwise.
    """
    num = int(sub.replace('sub-', ''))
    if num > 6000:
        return (BBHI_RECONALL / f"{sub}_ses-0{ses}_run-01").exists()
    return (SENIOR_RECONALL / f"{sub}_ses-0{ses}").exists()


def expected_dwi_files(sub, ses):
    """Return the DWI files expected for a subject/session.

    Args:
        sub (str): Subject ID in the format 'sub-xxx'.
        ses (int): Session number (1 or 2).

    Returns:
        list[Path]: Paths to the eddy-corrected image and the SIFT2 tractogram
        weights, or an empty list if the subject ID cannot be parsed.
    """
    try:
        num = int(sub.split('-')[1])
    except (IndexError, ValueError):
        return []
    base = BBHI_MRI / 'processed_data' if num > 5000 else SENIOR_MRI
    return [
        base / f"dtifit_ses-0{ses}_fsl-604/{sub}_ses-0{ses}/eddy_corrected_data.nii.gz",
        base / f"tracto_SIFT2/{sub}_ses-0{ses}/{sub}_ses-0{ses}_dwi_tractogram_10M_SIFT2_weights.txt",
    ]


def gather_dwi_checks(sub_ids, ses):
    """Check DWI file existence for each subject at a given session.

    Args:
        sub_ids (list[str]): Subject IDs in the format 'sub-xxx'.
        ses (int): Session number (1 or 2).

    Returns:
        dict: Maps each subject to {'eddy': bool, 'tract': bool}.
    """
    results = {}
    for sub in sub_ids:
        files = expected_dwi_files(sub, ses)
        if len(files) < 2:
            continue
        results[sub] = {'eddy': files[0].exists(), 'tract': files[1].exists()}
    return results


def expected_fmri_files(sub, ses):
    """Return the resting-state fMRI files expected for a subject/session.

    Args:
        sub (str): Subject ID in the format 'sub-xxx'.
        ses (int): Session number (1 or 2).

    Returns:
        list[Path]: Paths to the BOLD image in T1 space and the Schaefer/
        subcortical bold-space mask, or an empty list if the ID cannot be parsed.
    """
    try:
        num = int(sub.split('-')[1])
    except (IndexError, ValueError):
        return []
    if num > 5000:
        preproc = 'resting_preproc_fs6-recon' if ses == 1 else f'resting_preproc_fs6-recon_tp{ses}'
        bold = BBHI_MRI / 'processed_data' / preproc / sub / 'native_T1' / \
            f"{sub}_ses-0{ses}_run-01_rest_bold_ap_T1-space.nii.gz"
    else:
        bold = SENIOR_MRI / 'resting_preproc_fs6-recon' / sub / f'ses-0{ses}' / 'native_T1' / \
            f"{sub}_ses-0{ses}_run-01_rest_bold_ap_T1-space.nii.gz"
    mask = SCHAEFER_DIR / f"ses-0{ses}/{sub}/bold_space_masks/" \
        f"{sub}_ses-0{ses}_schaefer200_subcortical14_bold_space.nii.gz"
    return [bold, mask]


def gather_fmri_checks(sub_ids, ses):
    """Check resting-state fMRI file existence for each subject at a session.

    Args:
        sub_ids (list[str]): Subject IDs in the format 'sub-xxx'.
        ses (int): Session number (1 or 2).

    Returns:
        dict: Maps each subject to {'bold': bool, 'mask': bool}.
    """
    results = {}
    for sub in sub_ids:
        files = expected_fmri_files(sub, ses)
        if len(files) < 2:
            continue
        results[sub] = {'bold': files[0].exists(), 'mask': files[1].exists()}
    return results


def check_scrubbing(sub):
    """Check whether a subject was excluded for excessive fMRI head motion.

    Reads the timeseries-extraction log for the scrubbing exclusion marker.

    Args:
        sub (str): Subject ID in the format 'sub-xxx'.

    Returns:
        bool: True if the subject was scrubbed for excessive motion, else False.
    """
    if not SCRUBBING_LOG.exists():
        print(f"Warning: log file {SCRUBBING_LOG} not found")
        return False
    return f"Excluding {sub} due to excessive motion" in SCRUBBING_LOG.read_text()


def fix_year(date_str):
    """Expand a two-digit year (mm/dd/yy) to four digits.

    Mirrors the date handling in the data-cleaning notebooks.

    Args:
        date_str (str): Date string in 'm/d/yy' or 'm/d/yyyy' format.

    Returns:
        str: Date string with a four-digit year.
    """
    m, d, y = date_str.split('/')
    y = int(y)
    if y < 100:
        y += 2000 if y <= 25 else 1900
    return f"{m}/{d}/{y}"


def exact_age(birth, collect):
    """Compute exact age in years between two dates.

    Args:
        birth (str): Date of birth ('m/d/yy' or 'm/d/yyyy').
        collect (str): Assessment date ('m/d/yy' or 'm/d/yyyy').

    Returns:
        float: Age in years (days / 365.25).
    """
    b = datetime.strptime(fix_year(birth), "%m/%d/%Y")
    c = datetime.strptime(fix_year(collect), "%m/%d/%Y")
    return (c - b).days / 365.25


def clean_id(x):
    """Strip a 'sub-' prefix and return the bare ID string.

    Args:
        x: Subject identifier, with or without a 'sub-' prefix.

    Returns:
        str: The numeric ID as a string.
    """
    return str(x).replace('sub-', '')


def analysis_ids(sup):
    """Reproduce `analysis_ids` from analyses.R: the final SFC cohort.

    Subjects must be present in the cleaned table, both SFC files, and the
    weighted ROI file, with complete weighted SFC at both timepoints.

    Args:
        sup (pd.DataFrame): The cleaned superager table (must have an 'id' column).

    Returns:
        set[str]: Cleaned subject IDs in the final analysis cohort.
    """
    s1 = pd.read_csv(SFC_DIR / 'ses-01' / 'all_to_all_roi_matrices' / 'all_sfc_data_ses-01.csv')
    s2 = pd.read_csv(SFC_DIR / 'ses-02' / 'all_to_all_roi_matrices' / 'all_sfc_data_ses-02.csv')
    w = pd.read_csv(DATA_DIR / 'weighted_global_roi_averages.csv')

    def rename(c):
        c = re.sub(r'^w([12])_(.*)$', r'\2_\1', c)
        c = re.sub(r'^(.*)_tp([12])_(.*)$', r'\1_\3_\2', c)
        return c

    w.columns = [rename(c) for c in w.columns]
    wid = next(c for c in w.columns if c in ('subject_id', 'id'))
    w['idc'] = w[wid].map(clean_id)

    raw_ids = set(sup['id'].map(clean_id))
    s1_ids = set(s1.iloc[:, 0].map(clean_id))
    s2_ids = set(s2.iloc[:, 0].map(clean_id))
    in_all = raw_ids & s1_ids & s2_ids & set(w['idc'])

    key = ['sfc_weighted_mean_1', 'sfc_weighted_mean_2']
    complete = w[w['idc'].isin(in_all)]
    complete = set(complete[complete[key].notna().all(axis=1)]['idc'])
    return in_all & complete


def compute_ns():
    """Compute every participant count in the flow diagram.

    Returns:
        dict: Counts for each cohort box and each exclusion box.
    """
    # Total cohort and age eligibility (from raw, pre-cleaning sources) 
    main = pd.read_csv(RAW_DIR / 'BBHI Data Timept1 NPS.csv')
    main['w1_age'] = [exact_age(str(r['Q1 GNRL Age']), str(r['w1_nps_date']))
                      for _, r in main.iterrows()]
    main_total = len(main)
    main_under60 = int((main['w1_age'] < 60).sum())

    senior = pd.read_excel(RAW_DIR / 'bbhi senior data.xlsx')
    senior_total = len(senior)  # BBHI senior cohort all >=60

    total = main_total + senior_total
    n_excl_age = main_under60  
    n_eligible = total - n_excl_age

    # Follow-up attendance (across the full eligible cohort) 
    eligible_ids = (set(main.loc[main['w1_age'] >= 60, 'id'].map(clean_id))
                    | set(senior['ID'].map(clean_id)))
    tp2_main = set(pd.read_csv(RAW_DIR / 'BBHI Data Timept2 NPS.csv')['id'].map(clean_id))
    tp2_senior = set(pd.read_csv(RAW_DIR / 'Ministerio2024Wave2_DATA_2026-01-20_1245.csv')
                     ['record_id_np_w2'].map(clean_id))
    attended_ids = eligible_ids & (tp2_main | tp2_senior)
    n_attended = len(attended_ids)
    n_excl_followup = n_eligible - n_attended

    # Complete cognitive and demographic data at BOTH timepoints 
    sup = pd.read_csv(DATA_DIR / 'superager.csv')
    sup['idc'] = sup['id'].map(clean_id)
    cog_w1 = ['w1_delayed_recall_raw', 'w1_tmt_b_raw', 'w1_sem_fluency_raw',
              'w1_inverse_digits_raw', 'w1_ravlt_total']
    cog_w2 = ['w2_delayed_recall_raw', 'w2_tmt_b_raw', 'w2_sem_fluency_raw',
              'w2_inverse_digits_raw', 'w2_ravlt_total']
    demo = ['w1_age', 'YoE', 'sex']
    complete = sup[cog_w1 + cog_w2 + demo].notna().all(axis=1) & sup['w2_age'].notna()
    n_complete = int(complete.sum())
    n_excl_cog = n_attended - n_complete
    long_ids = [f"sub-{i}" for i in sup.loc[complete, 'idc']]

    # MRI data availability / quality
    # The final cohort is the structure-function-coupling sample (analysis_ids).
    # Remaining subjects are split into two exclusion reasons:
    #   * Incomplete or no MRI data: no scan, resting-state run <10 min (not
    #     processed, so no usable output), or missing structural/diffusion/fMRI output.
    #   * Excessive head motion: had complete usable MRI but excluded by scrubbing.
    analysis = analysis_ids(sup)
    dwi1 = gather_dwi_checks(long_ids, 1)
    dwi2 = gather_dwi_checks(long_ids, 2)
    fmri1 = gather_fmri_checks(long_ids, 1)
    fmri2 = gather_fmri_checks(long_ids, 2)

    def has_dwi(s):
        return (s in dwi1 and dwi1[s]['eddy'] and dwi1[s]['tract'] and
                s in dwi2 and dwi2[s]['eddy'] and dwi2[s]['tract'])

    def has_fmri(s):
        return (s in fmri1 and fmri1[s]['bold'] and fmri1[s]['mask'] and
                s in fmri2 and fmri2[s]['bold'] and fmri2[s]['mask'])

    def complete_mri(s):
        return (has_reconall(s, 1) and has_reconall(s, 2)
                and has_dwi(s) and has_fmri(s))

    n_final = n_excl_motion = n_excl_mridata = 0
    for s in long_ids:
        if clean_id(s) in analysis:
            n_final += 1
        elif complete_mri(s) and check_scrubbing(s):
            n_excl_motion += 1
        else:
            n_excl_mridata += 1
    n_mri_ok = n_complete - n_excl_mridata

    return {
        'total': total,
        'main_total': main_total,
        'senior_total': senior_total,
        'n_excl_age': n_excl_age,
        'n_eligible': n_eligible,
        'n_excl_followup': n_excl_followup,
        'n_attended': n_attended,
        'n_excl_cog': n_excl_cog,
        'n_complete': n_complete,
        'n_excl_mridata': n_excl_mridata,
        'n_mri_ok': n_mri_ok,
        'n_excl_motion': n_excl_motion,
        'n_final': n_final,
    }


def draw_consort(ns):
    """Draw and save the flow diagram.

    Six cohort boxes form the main vertical flow, with five exclusion boxes
    branching to the right at each step. Saved as a PNG in the figures folder.

    Args:
        ns (dict): Participant counts as returned by :func:`compute_ns`.
    """
    main_texts = [
        f"Total participants\n$n$ = {ns['total']}",
        f"Aged ≥60 at baseline\n$n$ = {ns['n_eligible']}",
        f"Completed follow-up assessment\n$n$ = {ns['n_attended']}",
        f"Complete cognitive and demographic\ndata at both timepoints\n$n$ = {ns['n_complete']}",
        f"Complete MRI data\n$n$ = {ns['n_mri_ok']}",
        f"Final analysis sample\n$n$ = {ns['n_final']}",
    ]
    excl_texts = [
        f"Excluded ($n$ = {ns['n_excl_age']}):\nAged <60 at baseline",
        f"Excluded ($n$ = {ns['n_excl_followup']}):\nNo follow-up visit",
        f"Excluded ($n$ = {ns['n_excl_cog']}):\nIncomplete cognitive or\ndemographic data",
        f"Excluded ($n$ = {ns['n_excl_mridata']}):\nIncomplete or no MRI\ndata available",
        f"Excluded ($n$ = {ns['n_excl_motion']}):\nExcessive head motion",
    ]

    n_main = len(main_texts)
    fig, ax = plt.subplots(figsize=(6.3, 7.0))
    ax.set_xlim(0.2, 6.9)
    ax.set_ylim(0, 8)
    ax.axis('off')

    main_x = 0.4    # left edge of main boxes
    main_w = 3.2    # main box width
    main_h = 0.66   # main box height
    excl_x = 4.0    # left edge of exclusion boxes
    excl_w = 2.7    # exclusion box width
    excl_h = 0.62   # exclusion box height

    top, bottom = 7.5, 0.5
    cx = main_x + main_w / 2
    main_cy = [top - i * (top - bottom) / (n_main - 1) for i in range(n_main)]

    box_style = dict(linewidth=0.8, edgecolor='black', facecolor='white')
    arrow_kw = dict(arrowstyle='->', color='black', lw=0.8)
    fontsize = 8.5

    for cy, text in zip(main_cy, main_texts):
        rect = mpatches.FancyBboxPatch((main_x, cy - main_h / 2), main_w, main_h,
                                       boxstyle='square,pad=0', **box_style)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
                multialignment='center')

    for i in range(n_main - 1):
        y_top = main_cy[i] - main_h / 2
        y_bot = main_cy[i + 1] + main_h / 2
        mid_y = (y_top + y_bot) / 2

        ax.annotate('', xy=(cx, y_bot), xytext=(cx, y_top), arrowprops=arrow_kw)
        ax.annotate('', xy=(excl_x, mid_y), xytext=(cx, mid_y), arrowprops=arrow_kw)

        rect = mpatches.FancyBboxPatch((excl_x, mid_y - excl_h / 2), excl_w, excl_h,
                                       boxstyle='square,pad=0', **box_style)
        ax.add_patch(rect)
        ax.text(excl_x + excl_w / 2, mid_y, excl_texts[i],
                ha='center', va='center', fontsize=fontsize, multialignment='center')

    out = Path(__file__).resolve().parent / 'figures' / 'consort_figure.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved to {out}")


if __name__ == '__main__':
    print("Computing participant numbers...")
    ns = compute_ns()
    for k, v in ns.items():
        print(f"  {k}: {v}")
    print("\nDrawing CONSORT diagram...")
    draw_consort(ns)
