from deepface import DeepFace
import cv2
import numpy as np
import pandas as pd

df = pd.read_csv("fairface_dataset.csv")  # Load your data here
df["predicted_race"] = None

for i, row in df.iterrows():
    try:
        analysis = DeepFace.analyze(
            img_path=row["image_path"],
            actions=['race'],
            enforce_detection=False
        )
        pred = analysis[0]["dominant_race"]
        df.loc[i, "predicted_race"] = pred

        print(f"{i}/{len(df)} → TRUE: {row['race']} | PRED: {pred}")

    except Exception as e:
        print("Error:", e)
        df.loc[i, "predicted_race"] = "error"
df.to_csv("fairface_results.csv", index=False)
