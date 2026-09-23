from pathlib import Path
import json
import joblib
import pandas as pd
from train_model import clean, FEATURES

ROOT = Path(__file__).parent
MODELS = ROOT / 'models'


def load():
    return joblib.load(MODELS / 'preprocessor.joblib'), joblib.load(MODELS / 'clf_crowd_level.joblib'), joblib.load(MODELS / 'reg_waiting_time.joblib')


def predict_row(row, date, slot, holiday=False, festival=False):
    pre, clf, reg = load()
    frame = pd.DataFrame([row])
    frame['Date'], frame['Hour'] = date, int(slot[:2])
    frame['Holiday'], frame['Special_Event_or_Festival'] = ('Yes' if holiday else 'No'), ('Yes' if festival else 'No')
    frame = clean(frame)
    x = pre.transform(frame[FEATURES])
    probabilities = clf.predict_proba(x)[0]
    level = clf.classes_[probabilities.argmax()]
    return level, float(reg.predict(x)[0]), {str(k): round(float(v), 2) for k, v in zip(clf.classes_, probabilities)}, float(probabilities.max())
