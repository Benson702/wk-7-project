import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Optional heavy imports (DeepFace) are imported only when used to avoid long import at startup
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False

st.set_page_config(page_title="FairVision — Facial Recognition Bias Audit", layout="wide")
st.title("FairVision — Facial Recognition Bias Audit (Demo)")

st.markdown("""
This Streamlit demo runs a facial-recognition fairness audit and visualizes bias metrics by protected group (race).
**Important:** Running DeepFace inside Streamlit may require significant CPU/GPU and proper dependencies.
If you don't have DeepFace installed or the model files, upload a `fairface_results.csv` produced by the offline audit.
""")

col1, col2 = st.columns([1,3])

with col1:
    st.header("Upload / Options")
    uploaded_csv = st.file_uploader("Upload `fairface_results.csv` (optional)", type=["csv"])
    run_deepface = st.checkbox("Run DeepFace prediction on uploaded images (slow)", value=False)
    st.write("")
    st.markdown("**Notes**")
    st.markdown("- If you upload `fairface_results.csv` (columns: image_path, race, predicted_race), the app will use it directly.")
    st.markdown("- To run DeepFace predictions from images, ensure DeepFace is installed in the environment and you have enough resources.")
    st.markdown("- Example dataset path (uploaded earlier): `/mnt/data/combined_notes.txt`")

with col2:
    st.header("Preview / Status")
    if uploaded_csv is None:
        st.info("No results CSV uploaded. You can upload a previously generated `fairface_results.csv` or enable DeepFace predictions.")
    else:
        st.success("Results CSV uploaded. Processing...")

# Load data
df = None
if uploaded_csv is not None:
    try:
        df = pd.read_csv('fairface_results.csv')
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")

# If user wants to run DeepFace on a folder of images (not recommended on Streamlit cloud),
# allow them to input a folder path on the server.
image_folder = st.text_input("Optional: server image folder path (use only if running locally)", value="")

if df is None and image_folder:
    # Build dataframe from folder
    rows = []
    for root, dirs, files in os.walk(image_folder):
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                # infer race from parent dir if dataset structured by race folders
                race = os.path.basename(root)
                rows.append({"image_path": os.path.join(root,f), "race": race})
    if rows:
        df = pd.DataFrame(rows)
        st.success(f"Discovered {len(df)} images under {image_folder}")

# If df exists and no predicted_race column, optionally run DeepFace to generate predictions
if df is not None and 'predicted_race' not in df.columns and run_deepface:
    if not DEEPFACE_AVAILABLE:
        st.error("DeepFace is not installed in this environment. Please install it and restart.")
    else:
        st.info("Running DeepFace race analysis (this will take time)...")
        preds = []
        for idx, row in df.iterrows():
            img = row.get("image_path")
            try:
                analysis = DeepFace.analyze(img_path=img, actions=['race'], enforce_detection=False)
                pred = analysis[0]['dominant_race']
            except Exception as e:
                pred = "error"
            preds.append(pred)
            if idx%50==0:
                st.write(f"Processed {idx+1}/{len(df)}")
        df['predicted_race'] = preds
        st.success("DeepFace predictions complete.")
        st.download_button("Download results CSV", data=df.to_csv(index=False), file_name="fairface_results.csv")

# If df has predicted results, compute fairness metrics
if df is not None and 'predicted_race' in df.columns:
    # clean
    df = df.dropna(subset=['race'])
    df['race'] = df['race'].astype(str).str.strip().str.lower()
    df['predicted_race'] = df['predicted_race'].astype(str).str.strip().str.lower()

    # compute per-group metrics: accuracy and selection rate
    groups = df['race'].unique().tolist()
    metrics = []
    for g in groups:
        sub = df[df['race']==g]
        acc = (sub['race'] == sub['predicted_race']).mean()
        selection_rate = (sub['predicted_race'] == g).mean()  # fraction predicted as this group
        metrics.append({'race':g, 'accuracy':acc, 'selection_rate':selection_rate, 'count':len(sub)})

    metrics_df = pd.DataFrame(metrics).sort_values('count', ascending=False)
    st.subheader("Fairness Metrics by Race")
    st.dataframe(metrics_df)

    # Visualizations
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    sns.barplot(x='race', y='accuracy', data=metrics_df, ax=ax[0])
    ax[0].set_title("Accuracy by Race")
    ax[0].set_xlabel("")
    ax[0].set_ylabel("Accuracy")
    ax[0].tick_params(axis='x', rotation=45)

    sns.barplot(x='race', y='selection_rate', data=metrics_df, ax=ax[1])
    ax[1].set_title("Selection Rate by Race (predicted as group)")
    ax[1].set_xlabel("")
    ax[1].set_ylabel("Selection Rate")
    ax[1].tick_params(axis='x', rotation=45)

    st.pyplot(fig)

    # Disparate impact: choose reference group as largest
    ref = metrics_df.loc[metrics_df['count'].idxmax(),'race']
    ref_rate = metrics_df.loc[metrics_df['race']==ref,'selection_rate'].values[0]
    metrics_df['disparate_impact'] = metrics_df['selection_rate'] / ref_rate
    st.subheader(f"Disparate Impact Ratios (reference: {ref})")
    st.dataframe(metrics_df[['race','count','selection_rate','disparate_impact']])

    # Provide guidance and ethics notes
    st.markdown("---")
    st.header("Ethical Interpretation & Recommendations")
    st.markdown("""
    **What this means:** Large differences in accuracy or selection rates across racial groups indicate the model treats groups differently — a sign of bias.
    **Recommendations:**
    - Balance training data across groups (data augmentation / collect more data).
    - Use fairness-aware training (reweighing, adversarial debiasing).
    - Add model transparency: report per-group metrics with any deployment.
    - Human-in-the-loop for high-stakes decisions; avoid automated-only actions.
    """)

else:
    st.info("No predicted results available. Upload `fairface_results.csv` or enable DeepFace predictions (if available).")

st.markdown("---")
st.caption("Project scaffold generated. Example uploaded file path from this session: /mnt/data/combined_notes.txt")
