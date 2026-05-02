# Tying the Data Knot 💞

**Love, Life & Likes — A Machine Learning Investigation**

A multi-page Streamlit application built for the WIA1006 / WID3006 Machine Learning group assignment, Semester 2 Session 2025/2026, Faculty of Computer Science & Information Technology, Universiti Malaya.

## What this project is

We were given a synthetic 50,000-row dating-app behaviour dataset and asked to build machine learning models to predict match outcomes. We trained five different classifiers — Logistic Regression, Decision Tree, Random Forest, SVM, and KNN — with hyperparameter tuning and 5-fold cross-validation, and discovered something more interesting than the prediction itself:

> **Every model converges to ~50% accuracy. That's chance level on a balanced binary task.**

Rather than treating that as a failure, we built this app to investigate *why*. Using mutual information analysis and PCA, we demonstrate that the dataset itself contains essentially no signal about match outcomes — the highest-scoring feature carries less than 0.005 bits of information about the target, where a useful predictor would carry 0.1–0.5 bits.

The honest conclusion of our project: **sometimes the most rigorous answer in machine learning is "the data doesn't support the question."**

## The app

Five pages, all themed in plum, sage, and warm white:

- **🏠 Home** — project intro, team, and the headline finding
- **📊 EDA Dashboard** — interactive exploration across demographics, behaviour, and outcomes
- **🔮 Predictor** — build a profile with sliders and dropdowns, get a live prediction (with an honest disclaimer)
- **⚖️ Model Comparison** — all five models side by side, with confusion matrices, hyperparameters, and cross-validation
- **🔬 Insights** — mutual information analysis, PCA visualisation, and the conceptual verdict

## How to run it locally

You'll need Python 3.9 or newer.

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/dating-ml-app.git
cd dating-ml-app

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # on Windows
# source venv/bin/activate   # on macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Re-run training from scratch — only needed if you change train.py
# The pre-built artifacts/ folder is included, so the app runs immediately.
python train.py

# 5. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Project structure
dating-ml-app/
├── app.py                          # Home page + entry point
├── train.py                        # One-shot training pipeline
├── utils.py                        # Shared loaders, theme, CSS
├── requirements.txt                # Pinned dependencies
├── pages/                          # Streamlit auto-discovers these
│   ├── 1_📊_EDA_Dashboard.py
│   ├── 2_🔮_Predictor.py
│   ├── 3_⚖️_Model_Comparison.py
│   └── 4_🔬_Insights.py
├── artifacts/                      # Saved models, scaler, encoders, metrics
│   ├── models.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── metadata.json
│   └── ...
├── .streamlit/
│   └── config.toml                 # Theme configuration
└── dating_app_behavior_dataset_extended1.csv

## Methodology in brief

1. **Preprocessing** — cleaned text encoding artifacts, one-hot expanded the multi-label `interest_tags` column into 49 binary columns, label-encoded categoricals with one encoder per column (saved for inverse transformations on the Predictor page).
2. **Target construction** — collapsed the 10 raw match outcomes into a binary target: Success (Relationship Formed, Date Happened, Mutual Match, Instant Match) vs Failure (everything else). Dataset rebalanced 50/50 by undersampling the majority class.
3. **Feature engineering** — added three binary flags summarising income bracket, education level, and relationship intent. Final feature count: 72.
4. **Modelling** — 5 models, GridSearchCV for hyperparameter tuning, 5-fold cross-validation, evaluation on a stratified hold-out test set.
5. **Diagnostics** — mutual information per feature, 2D PCA projection of the training set.

All five models scored within ~1 percentage point of each other on the test set and within ~0.3 percentage points of each other under cross-validation. The MI analysis confirmed that no feature carries meaningful information about the target.

## Team

**WIA1006 / WID3006 Machine Learning, Semester 2 Session 2025/2026**

- Ammar Kapadia *(Group Leader)*
- Tan Ker Li
- Yim Zi Hao
- Chew Chen Xi
- Ng Tze Fhung
- Muhaimin Afif Bin Mushahar

## Dataset

[Dating App Behavior Dataset on Kaggle](https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset) — synthetic, 50,000 rows, 19 base features.

## Disclaimer

The Predictor page is intentionally honest about its accuracy. This model is for academic demonstration purposes only and should not be taken as advice on dating app behaviour. As we demonstrate on the Insights page, the underlying dataset does not contain the kind of signal needed to genuinely predict match outcomes — and that's the finding of our project.