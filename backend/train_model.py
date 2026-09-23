import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from utils.slots import time_slot_for_hour

ROOT = Path(__file__).parent
DATA = ROOT / 'data' / 'crowd_data.csv'
MODELS = ROOT / 'models'
FEATURES = ['Location', 'Place_Type', 'Day', 'TimeSlot', 'WeekendFlag', 'HolidayFlag', 'FestivalFlag', 'Weather', 'Traffic', 'Parking_Percentage', 'Mobile_Density_Score', 'Booked_Visitors', 'Tokens_Issued', 'Month']
CATEGORICAL = ['Location', 'Place_Type', 'Day', 'TimeSlot', 'Weather', 'Traffic']
NUMERIC = ['WeekendFlag', 'HolidayFlag', 'FestivalFlag', 'Parking_Percentage', 'Mobile_Density_Score', 'Booked_Visitors', 'Tokens_Issued', 'Month']


def clean(df):
    df = df.copy()
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['TimeSlot'] = df['Hour'].astype(int).map(time_slot_for_hour)
    df['WeekendFlag'] = df['Weekend'].eq('Yes').astype(int)
    df['HolidayFlag'] = df['Holiday'].eq('Yes').astype(int)
    df['FestivalFlag'] = df['Special_Event_or_Festival'].eq('Yes').astype(int)
    df['Mobile_Density_Score'] = df['Mobile_Density'].map({'Low': 25, 'Medium': 60, 'High': 90}).fillna(60)
    df['Booked_Visitors'] = pd.to_numeric(df['Booked_Visitors'], errors='coerce').fillna(0)
    df['Tokens_Issued'] = pd.to_numeric(df['Tokens_Issued'], errors='coerce').fillna(0)
    df['Parking_Percentage'] = pd.to_numeric(df['Parking_Percentage'], errors='coerce').fillna(0)
    df['Temperature'] = df['Weather'].map({'Sunny': 31.0, 'Cloudy': 27.0, 'Rainy': 23.0}).fillna(27.0)
    df['WaitMinutes'] = df['Estimated_Waiting_Time'].astype(str).str.extract(r'(\d+(?:\.\d+)?)')[0].astype(float)
    df['WaitMinutes'] = df['WaitMinutes'].where(~df['Estimated_Waiting_Time'].astype(str).str.contains('hour'), df['WaitMinutes'] * 60)
    return df


def main():
    df = clean(pd.read_csv(DATA))
    x = df[FEATURES]
    y_class = df['Crowd_Level']
    y_reg = df['WaitMinutes']
    indices = range(len(df))
    train_idx, test_idx = train_test_split(list(indices), test_size=0.2, random_state=42, stratify=y_class)
    transformer = ColumnTransformer([('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)], remainder='passthrough')
    x_train = transformer.fit_transform(x.iloc[train_idx])
    x_test = transformer.transform(x.iloc[test_idx])
    clf = RandomForestClassifier(n_estimators=220, random_state=42, class_weight='balanced', n_jobs=-1)
    reg = RandomForestRegressor(n_estimators=220, random_state=42, n_jobs=-1)
    clf.fit(x_train, y_class.iloc[train_idx])
    reg.fit(x_train, y_reg.iloc[train_idx])
    pred_class = clf.predict(x_test)
    pred_wait = reg.predict(x_test)
    metrics = {'classifier': {'accuracy': accuracy_score(y_class.iloc[test_idx], pred_class), 'macro_f1': f1_score(y_class.iloc[test_idx], pred_class, average='macro')}, 'regressor': {'rmse': mean_squared_error(y_reg.iloc[test_idx], pred_wait) ** 0.5, 'mae': mean_absolute_error(y_reg.iloc[test_idx], pred_wait), 'r2': r2_score(y_reg.iloc[test_idx], pred_wait)}}
    MODELS.mkdir(exist_ok=True)
    joblib.dump(transformer, MODELS / 'preprocessor.joblib')
    joblib.dump(clf, MODELS / 'clf_crowd_level.joblib')
    joblib.dump(reg, MODELS / 'reg_waiting_time.joblib')
    names = list(transformer.get_feature_names_out())
    (MODELS / 'feature_names.json').write_text(json.dumps(names, indent=2))
    fallback = {'place': df.groupby('Location')[['Parking_Percentage', 'Mobile_Density_Score']].mean().round(2).to_dict('index'), 'global': {'Parking_Percentage': float(df.Parking_Percentage.mean()), 'Mobile_Density_Score': float(df.Mobile_Density_Score.mean())}}
    (MODELS / 'fallback_tables.json').write_text(json.dumps(fallback, indent=2))
    metadata = {'rows': len(df), 'metrics': metrics, 'classes': list(clf.classes_), 'top_classifier_features': sorted(zip(names, clf.feature_importances_), key=lambda x: x[1], reverse=True)[:15], 'top_regressor_features': sorted(zip(names, reg.feature_importances_), key=lambda x: x[1], reverse=True)[:15]}
    (MODELS / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=float))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
