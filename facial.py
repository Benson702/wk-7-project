import os
import pandas as pd

DATA_DIR = r"FairFace/train"

rows = []

for race_folder in os.listdir(DATA_DIR):
    full_path = os.path.join(DATA_DIR, race_folder)
    if os.path.isdir(full_path):
        for img_name in os.listdir(full_path):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({
                    "image_path": os.path.join(full_path, img_name),
                    "race": race_folder
                })

df = pd.DataFrame(rows)
df.to_csv("fairface_dataset.csv", index=False)

print("Dataset created with", len(df), "images")
print(df.head())
