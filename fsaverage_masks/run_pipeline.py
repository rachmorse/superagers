#!/usr/bin/env python3

import subprocess
import sys

# List of scripts to run in order
scripts = [
    "01_fsaverage_to_t1.py",
    "02_subcortical_to_t1.py",
    "03_combine_t1_altases.py",
    "04_t1_to_dwi_bold.py"
]

# Run each script and stop if any fail
for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run([sys.executable, script])
    
    if result.returncode != 0:
        print(f"Error: {script} failed with exit code {result.returncode}")
        sys.exit(1)  # Exit with error code
    
    print(f"Successfully completed {script}")
    print("-" * 30)

print("All scripts completed successfully.")