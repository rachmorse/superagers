import pandas as pd
from pathlib import Path
from typing import Union

def save_group_averages(group_df, group_name, output_file):
    """Compute the means for a group, drop non-relevant columns, and save to CSV.
    
    Args:
        group_df (pd.DataFrame): DataFrame containing the group data.
        group_name (str): Name of the group.
        output_file (Union[str, Path]): Path to save the output CSV file.
    """
    mean_vals = group_df.mean(numeric_only=True)
    
    # Prepare and save the result DataFrame
    result_df = pd.DataFrame([mean_vals], index=[group_name])
    result_df = result_df.drop(columns=['superager', 'maintainer'], errors='ignore')
    result_df.index.name = 'id'
    
    # Print the output path
    print(f"Saving results to {output_file}")
    
    # Save to CSV
    result_df.to_csv(output_file)

def process_connectivity(connectivity_file: Union[str, Path], superager_file: Union[str, Path], output_files: dict):
    """Process and merge connectivity data with superager and maintainer status, then calculate averages.

    Args:
        connectivity_file (Union[str, Path]): Path to the connectivity CSV file.
        superager_file (Union[str, Path]): Path to the superager status CSV file.
        output_files (dict): Dictionary to save averages for each category.
    """
    # Load the data
    df_connectivity = pd.read_csv(connectivity_file)
    df_superager = pd.read_csv(superager_file)

    # Ensure necessary columns are present
    df_superager = df_superager[['id', 'superager', 'maintainer']]

    # Ensure 'id' columns have the same data type
    df_connectivity['id'] = df_connectivity['id'].astype(str)
    df_superager['id'] = df_superager['id'].astype(str).apply(lambda x: 'sub-' + x) 

    # Merge dataframes on 'id'
    df = pd.merge(df_connectivity, df_superager, on='id', how="inner")

    # Process individual groups
    for column, prefix in [('superager', 'superagers'), ('maintainer', 'maintainers')]:
        for label, group_df in df.groupby(column):
            if label == 1:
                group_name = f"{prefix}"
            else:
                group_name = f"non_{prefix}" if column == 'superager' else "decliners"
            output_file = output_files[group_name]
            save_group_averages(group_df, group_name, output_file)

    # Process combined groups
    for (superager_label, maintainer_label), group_df in df.groupby(['superager', 'maintainer']):
        if superager_label == 1 and maintainer_label == 1:
            group_name = 'superager_maintainers'
        elif superager_label == 1 and maintainer_label == 0:
            group_name = 'superager_decliners'
        elif superager_label == 0 and maintainer_label == 1:
            group_name = 'non_superager_maintainers'
        else:
            group_name = 'non_superager_decliners'

        output_file = output_files[group_name]
        save_group_averages(group_df, group_name, output_file)

    print("CSV files created successfully!")

def main(output_dir: Union[str, Path], connectivity_file: Union[str, Path], superager_file: Union[str, Path]):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output file paths for different categories
    output_files = {
        'superagers': output_dir / "superagers_average.csv",
        'non_superagers': output_dir / "non_superagers_average.csv",
        'maintainers': output_dir / "maintainers_average.csv",
        'decliners': output_dir / "decliners_average.csv",
        'superager_maintainers': output_dir / "superager_maintainers_average.csv",
        'superager_decliners': output_dir / "superager_decliners_average.csv",
        'non_superager_maintainers': output_dir / "non_superager_maintainers_average.csv",
        'non_superager_decliners': output_dir / "non_superager_decliners_average.csv",
    }
    
    process_connectivity(connectivity_file, superager_file, output_files)

if __name__ == "__main__":
    ses = "02"
    superager_file = "/home/rachel/Desktop/data/maintainer_superager_data.csv"  
    output_directory = Path(f"/home/rachel/Desktop/schaefer_analysis/connectivity_matrices/ses-{ses}/all_to_all_roi_matrices")
    connectivity_file = Path(f"{output_directory}/fisher_z_all_to_all_roi_matrix.csv")

    main(
        output_dir=output_directory,
        connectivity_file=connectivity_file,
        superager_file=superager_file,
    )