"""
train.py — One-shot training pipeline for the Dating ML App.

Run this ONCE from your terminal:
    python train.py

It loads the CSV, preprocesses, trains 5 models with hyperparameter tuning,
computes evaluation metrics + cross-validation + mutual information + PCA,
and saves everything into ./artifacts/ for the Streamlit app to load.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
CSV_PATH = "dating_app_behavior_dataset_extended1.csv"
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

SUCCESS_OUTCOMES = ["Relationship Formed", "Date Happened", "Mutual Match", "Instant Match"]
HIGH_INCOME = ["High", "Very High", "Upper-Middle", "Middle"]
HIGH_EDU = ["Bachelor's", "Master's", "PhD", "Postdoc"]
HIGH_INTENT = ["Serious Relationship", "Casual Dating"]


# --------------------------------------------------------------------------
# 1. LOAD + CLEAN
# --------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading and cleaning data")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

# Clean weird apostrophes from CSV encoding issues
for col in ["education_level", "income_bracket", "relationship_intent"]:
    df[col] = df[col].astype(str).str.replace("\u2019", "'").str.replace("â€™", "'")

# Save raw class distribution for EDA page
class_distribution = df["match_outcome"].value_counts().to_dict()

# One-hot encode interest_tags (multi-label column)
df["interest_tags"] = df["interest_tags"].astype(str)
interests_split = df["interest_tags"].str.get_dummies(sep=", ")
interest_columns = list(interests_split.columns)
df = pd.concat([df, interests_split], axis=1).drop("interest_tags", axis=1)
print(f"Split interest_tags into {len(interest_columns)} binary columns")


# --------------------------------------------------------------------------
# 2. BUILD TARGET + ENGINEERED FEATURES
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Building target + engineered features")
print("=" * 60)

df["target_success"] = df["match_outcome"].apply(lambda x: 1 if x in SUCCESS_OUTCOMES else 0)
df["feat_income_high"] = df["income_bracket"].apply(lambda x: 1 if x in HIGH_INCOME else 0)
df["feat_edu_high"] = df["education_level"].apply(lambda x: 1 if x in HIGH_EDU else 0)
df["feat_intent_high"] = df["relationship_intent"].apply(lambda x: 1 if x in HIGH_INTENT else 0)

print(f"Class balance (raw): "
      f"Success={int((df['target_success']==1).sum()):,}, "
      f"Failure={int((df['target_success']==0).sum()):,}")


# --------------------------------------------------------------------------
# 3. ENCODE CATEGORICALS (one encoder per column, all saved)
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Encoding categorical columns")
print("=" * 60)

# Columns we'll keep as categorical features (encoded)
keep_categorical = [
    "gender", "sexual_orientation", "location_type",
    "app_usage_time_label", "swipe_right_label", "swipe_time_of_day",
    "zodiac_sign", "body_type",
]

# Drop the original text columns we replaced with feat_* engineered ones
# (we keep their engineered binary versions instead)
text_cols_to_drop = ["match_outcome", "relationship_intent",
                     "income_bracket", "education_level"]

encoders = {}
for col in keep_categorical:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"  Encoded '{col}': {len(le.classes_)} unique values")


# --------------------------------------------------------------------------
# 4. BALANCE + SPLIT
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Balancing + train/test split")
print("=" * 60)

wins = df[df["target_success"] == 1]
losses = df[df["target_success"] == 0]
n = min(len(wins), len(losses))
df_balanced = pd.concat([
    wins.sample(n=n, random_state=RANDOM_STATE),
    losses.sample(n=n, random_state=RANDOM_STATE),
]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Balanced dataset: {len(df_balanced):,} rows ({n:,} per class)")

X = df_balanced.drop(columns=text_cols_to_drop + ["target_success"], errors="ignore")
y = df_balanced["target_success"]
feature_names = list(X.columns)
print(f"Feature count: {len(feature_names)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")


# --------------------------------------------------------------------------
# 5. MUTUAL INFORMATION (the smoking gun for "why ~50%?")
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Computing mutual information (feature <-> target)")
print("=" * 60)

mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=RANDOM_STATE)
mi_df = pd.DataFrame({"feature": feature_names, "mi_score": mi_scores}) \
          .sort_values("mi_score", ascending=False).reset_index(drop=True)
print(mi_df.head(10).to_string(index=False))
print(f"\nMax MI: {mi_df['mi_score'].max():.4f}  |  "
      f"Mean MI: {mi_df['mi_score'].mean():.4f}")
print("(Near-zero MI across all features = features genuinely don't predict target)")


# --------------------------------------------------------------------------
# 6. PCA (for visualisation on Insights page)
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: PCA 2D projection")
print("=" * 60)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")


# --------------------------------------------------------------------------
# 7. TRAIN 5 MODELS WITH LIGHT HYPERPARAMETER TUNING
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: Training 5 models")
print("=" * 60)

# Subsample for SVM and grid search (full balanced set is ~28k; SVM is O(n^2))
SVM_SAMPLE = min(5000, len(X_train_scaled))
rng = np.random.RandomState(RANDOM_STATE)
svm_idx = rng.choice(len(X_train_scaled), SVM_SAMPLE, replace=False)
X_train_svm = X_train_scaled[svm_idx]
y_train_svm = y_train.iloc[svm_idx]

model_specs = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "param_grid": {"C": [0.1, 1.0, 10.0]},
        "X_train": X_train_scaled, "y_train": y_train,
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "param_grid": {"max_depth": [5, 10, 20, None]},
        "X_train": X_train_scaled, "y_train": y_train,
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": {"n_estimators": [100, 200], "max_depth": [10, 20]},
        "X_train": X_train_scaled, "y_train": y_train,
    },
    "SVM": {
        "estimator": SVC(random_state=RANDOM_STATE),
        "param_grid": {"C": [0.1, 1.0], "kernel": ["linear", "rbf"]},
        "X_train": X_train_svm, "y_train": y_train_svm,
    },
    "KNN": {
        "estimator": KNeighborsClassifier(n_jobs=-1),
        "param_grid": {"n_neighbors": [5, 15, 31]},
        "X_train": X_train_scaled, "y_train": y_train,
    },
}

trained_models = {}
metrics = {}
for name, spec in model_specs.items():
    print(f"\n--- {name} ---")
    grid = GridSearchCV(
        spec["estimator"], spec["param_grid"],
        cv=3, scoring="accuracy", n_jobs=-1
    )
    grid.fit(spec["X_train"], spec["y_train"])
    best_model = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")

    # Test set metrics
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Cross-validation
    cv_scores = cross_val_score(best_model, spec["X_train"], spec["y_train"],
                                cv=CV_FOLDS, scoring="accuracy", n_jobs=-1)

    print(f"  Test accuracy: {acc:.4f}  |  CV mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    trained_models[name] = best_model
    metrics[name] = {
        "best_params": grid.best_params_,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "cv_scores": cv_scores.tolist(),
    }


# --------------------------------------------------------------------------
# 8. SAVE EVERYTHING
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8: Saving artifacts")
print("=" * 60)

# Pick best model by test accuracy
best_name = max(metrics, key=lambda k: metrics[k]["accuracy"])
print(f"Best model: {best_name} ({metrics[best_name]['accuracy']:.4f})")

joblib.dump(trained_models, ARTIFACTS_DIR / "models.pkl")
joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
joblib.dump(encoders, ARTIFACTS_DIR / "encoders.pkl")
joblib.dump(pca, ARTIFACTS_DIR / "pca.pkl")

np.save(ARTIFACTS_DIR / "X_train_pca.npy", X_train_pca)
np.save(ARTIFACTS_DIR / "y_train.npy", y_train.values)

mi_df.to_csv(ARTIFACTS_DIR / "mutual_information.csv", index=False)

# Save a small EDA-friendly sample of the cleaned data (full df is big)
eda_sample = df.sample(n=min(10000, len(df)), random_state=RANDOM_STATE)
eda_sample.to_parquet(ARTIFACTS_DIR / "eda_sample.parquet", index=False)

with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
    json.dump({
        "feature_names": feature_names,
        "interest_columns": interest_columns,
        "categorical_encoded": keep_categorical,
        "engineered_features": ["feat_income_high", "feat_edu_high", "feat_intent_high"],
        "best_model": best_name,
        "metrics": metrics,
        "class_distribution": class_distribution,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "success_outcomes": SUCCESS_OUTCOMES,
    }, f, indent=2)

print(f"\nAll artifacts saved to: {ARTIFACTS_DIR.resolve()}")
print("Done. You can now run the Streamlit app.")