from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, classification_report

BASE = Path(__file__).resolve().parent
DATA = BASE / "enhanced_smart_crowd_dataset_realistic_2025.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Date", "Location", "Hour"]).reset_index(drop=True)

FEATURES = [
    "Location","Place_Type","Day","Hour","Weekend","Holiday",
    "Special_Event_or_Festival","Weather","Temperature","Traffic_Level",
    "Parking_Occupancy","Mobile_Density","Historical_Average_Visitors"
]
CATEGORICAL = ["Location","Place_Type","Day","Weather","Traffic_Level","Mobile_Density"]
NUMERIC = ["Hour","Weekend","Holiday","Special_Event_or_Festival","Temperature",
           "Parking_Occupancy","Historical_Average_Visitors"]

for c in ["Weekend","Holiday","Special_Event_or_Festival"]:
    df[c] = df[c].map({"Yes":1,"No":0}).astype(int)

# Time-based validation: train on the past, test on later dates.
train = df[df["Date"] < "2025-11-01"].copy()
test = df[df["Date"] >= "2025-11-01"].copy()

def make_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])

classifier = Pipeline([
    ("prep", make_preprocessor()),
    ("model", RandomForestClassifier(
        n_estimators=160,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])

wait_model = Pipeline([
    ("prep", make_preprocessor()),
    ("model", RandomForestRegressor(
        n_estimators=120,
        min_samples_leaf=6,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ))
])

classifier.fit(train[FEATURES], train["Crowd_Level"])
wait_model.fit(train[FEATURES], train["Waiting_Time"])

pred = classifier.predict(test[FEATURES])
wait_pred = np.clip(wait_model.predict(test[FEATURES]), 5, 300)

metrics = {
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "test_period": "2025-11-01 to 2025-12-31",
    "accuracy": float(accuracy_score(test["Crowd_Level"], pred)),
    "macro_f1": float(f1_score(test["Crowd_Level"], pred, average="macro")),
    "weighted_f1": float(f1_score(test["Crowd_Level"], pred, average="weighted")),
    "waiting_mae_minutes": float(mean_absolute_error(test["Waiting_Time"], wait_pred)),
    "waiting_rmse_minutes": float(np.sqrt(mean_squared_error(test["Waiting_Time"], wait_pred)))
}

print("\n=== VALIDATION ===")
print(json.dumps(metrics, indent=2))
print("\nClassification report:")
print(classification_report(test["Crowd_Level"], pred, digits=3))

joblib.dump(classifier, MODEL_DIR / "crowd_classifier.joblib")
joblib.dump(wait_model, MODEL_DIR / "waiting_regressor.joblib")
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

check = test[["Location","Date","Hour","Crowd_Level","Waiting_Time"]].copy()
check["Predicted_Crowd"] = pred
check["Predicted_Wait"] = np.round(wait_pred, 1)
check.to_csv(MODEL_DIR / "validation_sample.csv", index=False)

print("\nModels saved in backend/models/")
