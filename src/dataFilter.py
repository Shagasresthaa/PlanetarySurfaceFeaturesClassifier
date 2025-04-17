import os
import shutil
import pandas as pd

LABEL_FILE = 'data/hirise-map-proj-v3/labels-map-proj-v3.txt'
IMAGE_DIR = 'data/hirise-map-proj-v3/map-proj-v3/'
OUTPUT_DIR = 'data/filterData/'
CSV_OUTPUT = 'data/index.csv'

# Labels to keep for training the model
label_map = {
    1: 'crater',
    2: 'dark dune',
    4: 'bright dune',
    5: 'impact ejecta'
}

# Create output directory and load the original project data index label file
os.makedirs(OUTPUT_DIR, exist_ok=True)
filtered_data = []

with open(LABEL_FILE, 'r') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        filename, label_str = line.split()
        label = int(label_str)

        if label in label_map:
            src_path = os.path.join(IMAGE_DIR, filename)
            dst_path = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                filtered_data.append({
                    'filename': filename,
                    'label_id': label,
                    'label_name': label_map[label]
                })
            else:
                print(f"[WARN] File not found: {src_path}")

# Save new index to CSV
df = pd.DataFrame(filtered_data)
df.to_csv(CSV_OUTPUT, index=False)
print(f"[INFO] Saved {len(df)} entries to {CSV_OUTPUT}")
