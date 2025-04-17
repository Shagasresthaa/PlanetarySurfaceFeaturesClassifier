import pandas as pd
import os
import shutil

INDEX_CSV = 'data/index.csv'
IMAGE_FOLDER = 'data/filterData'
OUTPUT_CSV = 'data/balancedDataset.csv'
OUTPUT_FOLDER = 'data/finalDataset'
TARGET_CRATER_COUNT = 1200

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
df = pd.read_csv(INDEX_CSV)

# Fix Label Names
df['label_name'] = df['label_name'].str.strip().str.replace(' ', '_')

# Downsample crater class
crater_df = df[df['label_name'] == 'crater']
other_df = df[df['label_name'] != 'crater']
crater_sampled = crater_df.sample(n=TARGET_CRATER_COUNT, random_state=42)
balanced_df = pd.concat([crater_sampled, other_df]).sort_values(by='filename')
balanced_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved dataset with {len(balanced_df)} entries to {OUTPUT_CSV}")

for _, row in balanced_df.iterrows():
    fname = row['filename']
    label = row['label_name']
    class_folder = os.path.join(OUTPUT_FOLDER, label)
    os.makedirs(class_folder, exist_ok=True)

    src = os.path.join(IMAGE_FOLDER, fname)
    dst = os.path.join(class_folder, fname)

    if os.path.exists(src):
        shutil.copy2(src, dst)
    else:
        print(f"File missing: {src}")
