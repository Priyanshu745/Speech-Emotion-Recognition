import pandas as pd
import os
import shutil

# Load the CSV file
csv_path = "testing_dataset.csv"
df = pd.read_csv(csv_path)

# Column that containsthe audio file paths
file_column = "filepath"

# Destination directory
output_dir = "testing"
os.makedirs(output_dir, exist_ok=True)

# Copy files
missing_files = []
for file_path in df[file_column]:
    if os.path.exists(file_path):
        shutil.copy(file_path, os.path.join(output_dir, os.path.basename(file_path)))
    else:
        missing_files.append(file_path)

print(f"✅ Copied {len(df) - len(missing_files)} files to '{output_dir}/'")
if missing_files:
    print("\n⚠️ Missing files:")
    for f in missing_files:
        print(" -", f)
