# FairVision — Facial Recognition Bias Audit (Streamlit)

This repository contains a Streamlit demo app that visualizes fairness metrics for facial-recognition outputs (race predictions).
It is a scaffold for your AI Ethics project: run an offline audit (generate `fairface_results.csv`) and upload the CSV to the app,
or optionally point the app at a server image folder and run DeepFace predictions (requires heavy compute).

## Files
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies

## How to run locally
1. Create a virtual environment (recommended)
2. Install requirements:
```bash
pip install -r requirements.txt
