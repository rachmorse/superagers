import pandas as pd
from pathlib import Path
from typing import Union

def process_connectivity(connectivity_file: Union[str, Path], superager_file: Union[str, Path], output_file_superagers: Union[str, Path], output_file_non_superagers: Union[str, Path]):
    """Process and merge connectivity data with superager status, then calculate averages.

    Args:
        connectivity_file (Union[str, Path]): Path to the connectivity CSV file.
        superager_file (Union[str, Path]): Path to the superager status CSV file.
        output_file_superagers (Union[str, Path]): Path to save averages for 'superagers'.
        output_file_non_superagers (Union[str, Path]): Path to save averages for 'non-superagers'.
    """
    # Load the data
    df_connectivity = pd.read_csv(connectivity_file)
    df_superager = pd.read_csv(superager_file)

    # Filter superager df to necessary columns
    # df_superager = df_superager[['id', 'superager']]
    df_superager = df_superager[['id', 'maintainer']]

    # Ensure 'id' columns have the same data type
    df_connectivity['id'] = df_connectivity['id'].astype(str)
    df_superager['id'] = df_superager['id'].astype(str).apply(lambda x: 'sub-' + x) 

    # Merge dataframes on 'id'
    df = pd.merge(df_connectivity, df_superager, on='id', how="inner")

    # Process each group (superager and non-superager)
    # for label, group_df in df.groupby('superager'):
    for label, group_df in df.groupby('maintainer'):
        # group_name = 'superager' if label == 1 else 'non-superager'
        group_name = 'maintainer' if label == 1 else 'decliner'

        # Compute means
        mean_vals = group_df.mean(numeric_only=True)
                
        # Prepare and save the result DataFrame
        result_df = pd.DataFrame([mean_vals], index=[group_name])
        # result_df = result_df.drop(columns='superager')
        result_df = result_df.drop(columns='maintainer')
        result_df.index.name = 'id'

        # Determine output file path
        output_file = output_file_superagers if label == 1 else output_file_non_superagers
        
        # Print the output path
        print(f"Saving results to {output_file}")
        
        # Save to CSV
        result_df.to_csv(output_file)

    print("CSV files created successfully!")

def main(output_dir: Union[str, Path], connectivity_file: Union[str, Path], superager_file: Union[str, Path]):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # output_file_superagers = output_dir / "superagers_average.csv"
    output_file_superagers = output_dir / "maintainers_average.csv"
    # output_file_non_superagers = output_dir / "non_superagers_average.csv"
    output_file_non_superagers = output_dir / "decliners_average.csv"

    process_connectivity(connectivity_file, superager_file, output_file_superagers, output_file_non_superagers)

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