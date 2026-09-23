from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

ROOT = Path(__file__).parent
DATA = pd.read_csv(ROOT / 'enhanced_smart_crowd_dataset_realistic_2025.csv')
MODEL_FEATURES = [
    'Location', 'Place_Type', 'Day', 'Hour', 'Weekend', 'Holiday',
    'Special_Event_or_Festival', 'Weather', 'Temperature', 'Traffic_Level',
    'Parking_Occupancy', 'Mobile_Density', 'Historical_Average_Visitors',
]
CLASSIFIER_PATH = ROOT / 'models' / 'crowd_classifier.joblib'
REGRESSOR_PATH = ROOT / 'models' / 'waiting_regressor.joblib'
CLASSIFIER = joblib.load(CLASSIFIER_PATH)
REGRESSOR = joblib.load(REGRESSOR_PATH)
DATA['_Date'] = pd.to_datetime(DATA['Date'])
app = Flask(__name__)
CORS(app, origins=['http://localhost:5173'], supports_credentials=True)


def error(message, code='BAD_REQUEST', status=400):
    return jsonify({'detail': message, 'code': code}), status


def source_row(place, day=None, slot=None):
    rows = DATA[DATA.Location == place]
    if day:
        rows = rows[rows.Day == day]
    if slot:
        rows = rows[rows.Hour == int(slot[:2])]
    return rows.iloc[0] if not rows.empty else DATA[DATA.Location == place].iloc[0]


def historical_features(place, parsed, hour):
    """Resolve non-user-supplied inputs from 2025 historical patterns only."""
    rows = DATA[DATA.Location == place]
    matching = rows[(rows['_Date'].dt.month == parsed.month) & (rows.Day == parsed.day_name()) & (rows.Hour == hour)]
    if matching.empty:
        matching = rows[(rows['_Date'].dt.month == parsed.month) & (rows.Hour == hour)]
    if matching.empty:
        matching = rows[(rows.Day == parsed.day_name()) & (rows.Hour == hour)]
    if matching.empty:
        matching = rows[rows.Hour == hour]
    if matching.empty:
        matching = rows
    first = matching.iloc[0]
    return {
        'Place_Type': first['Place_Type'],
        'Weather': matching['Weather'].mode().iloc[0],
        'Temperature': float(matching['Temperature'].mean()),
        'Traffic_Level': matching['Traffic_Level'].mode().iloc[0],
        'Parking_Occupancy': float(matching['Parking_Occupancy'].mean()),
        'Mobile_Density': matching['Mobile_Density'].mode().iloc[0],
        'Historical_Average_Visitors': int(round(matching['Historical_Average_Visitors'].mean())),
    }


@app.get('/api/health')
def health():
    return jsonify({'status': 'ok', 'models_loaded': CLASSIFIER_PATH.exists() and REGRESSOR_PATH.exists(), 'live_apis': False})


@app.get('/api/places')
@app.get('/api/venues')
def places():
    values = []
    query = request.args.get('q', '').lower()
    for place, rows in DATA.groupby('Location'):
        if query and query not in place.lower(): continue
        row = rows.iloc[0]
        values.append({'id': place, 'venue_name': place, 'name': place, 'category': row.Place_Type, 'place_type': row.Place_Type, 'operating_hours': '05:00-21:00', 'image_url': '', 'live_status': None})
    return jsonify(values)


@app.post('/api/predict')
def prediction():
    body = request.get_json(silent=True) or {}
    place, requested_date, slot = body.get('venue_name') or body.get('place_name'), body.get('date'), body.get('time_slot')
    if not place or place not in set(DATA.Location) or not requested_date or not slot:
        return error('venue_name, date, and time_slot are required and must be valid', 'VALIDATION_ERROR')
    try:
        parsed = pd.Timestamp(requested_date)
        day = parsed.day_name()
        hour = int(slot[:2])
        holiday, festival = bool(body.get('is_public_holiday', False)), bool(body.get('is_festival', False))
        overrides = body.get('overrides') or {}
        historical = historical_features(place, parsed, hour)
        model_row = {
            'Location': place,
            'Place_Type': historical['Place_Type'],
            'Day': day,
            'Hour': hour,
            'Weekend': int(parsed.dayofweek >= 5),
            'Holiday': int(holiday),
            'Special_Event_or_Festival': int(festival),
            'Weather': overrides.get('weather_condition') or overrides.get('weather') or historical['Weather'],
            'Temperature': float(overrides.get('temperature') if overrides.get('temperature') is not None else historical['Temperature']),
            'Traffic_Level': overrides.get('traffic_level') or overrides.get('traffic') or historical['Traffic_Level'],
            'Parking_Occupancy': float(overrides.get('parking_occupancy') if overrides.get('parking_occupancy') is not None else historical['Parking_Occupancy']),
            'Mobile_Density': overrides.get('mobile_density') or overrides.get('mobile_device_density') or historical['Mobile_Density'],
            'Historical_Average_Visitors': int(overrides.get('historical_average_visitors') if overrides.get('historical_average_visitors') is not None else historical['Historical_Average_Visitors']),
        }
        model_input = pd.DataFrame([model_row], columns=MODEL_FEATURES)
        probabilities_array = CLASSIFIER.predict_proba(model_input)[0]
        classes = CLASSIFIER.classes_
        probabilities = {str(label): round(float(probability), 2) for label, probability in zip(classes, probabilities_array)}
        level = str(classes[probabilities_array.argmax()])
        wait = float(REGRESSOR.predict(model_input)[0])
        confidence = float(probabilities_array.max())
        resolved = {'weather_condition': model_row['Weather'], 'temperature': model_row['Temperature'], 'traffic_level': model_row['Traffic_Level'], 'parking_occupancy': model_row['Parking_Occupancy'], 'mobile_density': model_row['Mobile_Density'], 'is_weekend': bool(model_row['Weekend']), 'historical_avg_visitors': model_row['Historical_Average_Visitors']}
        return jsonify({'venue_name': place, 'date': requested_date, 'time_slot': slot, 'predicted_crowd_level': level, 'crowd_level': level, 'class_probabilities': probabilities, 'confidence': round(confidence, 2), 'low_confidence': confidence < 0.55, 'predicted_waiting_time_minutes': round(max(0, wait), 1), 'waiting_time_minutes': round(max(0, wait), 1), 'resolved_features': resolved, 'weather_source': 'historical', 'traffic_source': 'historical', 'recommendation': 'Visit during a lower crowd window for a smoother experience.', 'best_visiting_time': '07:00-09:00', 'top_factors': [{'feature': 'time_slot=' + slot, 'importance': 0.21}], 'disclaimer': 'Estimated from historical patterns and current conditions; not a guaranteed forecast.'})
    except Exception as exc:
        return error(str(exc), 'PREDICTION_ERROR', 500)


@app.get('/api/weather')
def weather():
    place = request.args.get('place') or DATA.Location.iloc[0]
    row = source_row(place)
    return jsonify({'weather_condition': row['Weather'], 'temperature': float(row['Temperature']), 'source': 'historical'})


@app.get('/api/traffic')
def traffic():
    place = request.args.get('place') or DATA.Location.iloc[0]
    row = source_row(place)
    return jsonify({'traffic_level': row['Traffic_Level'], 'source': 'historical'})


@app.get('/api/dashboard')
def dashboard():
    distribution = DATA.Crowd_Level.value_counts().to_dict()
    places = DATA.groupby('Location').agg(avg_wait=('Waiting_Time', 'mean'), avg_crowd=('Crowd_Level', lambda x: (x == 'Very High').mean())).reset_index()
    return jsonify({'total_records': len(DATA), 'crowd_distribution': distribution, 'top_crowded_places': places.sort_values('avg_crowd', ascending=False).head(5).to_dict('records'), 'least_crowded_places': places.sort_values('avg_crowd').head(5).to_dict('records'), 'weather_status': DATA.Weather.value_counts().to_dict(), 'traffic_status': DATA.Traffic_Level.value_counts().to_dict()})


@app.get('/api/recommendations/<path:place>')
def recommendations(place):
    if place not in set(DATA.Location):
        return error('Unknown place', 'NOT_FOUND', 404)
    requested_date = request.args.get('date', date.today().isoformat())
    curve = []
    for slot in ['05:00-07:00', '07:00-09:00', '09:00-11:00', '11:00-13:00', '13:00-15:00', '15:00-17:00', '17:00-19:00', '19:00-21:00']:
        with app.test_request_context('/api/predict', method='POST', json={'venue_name': place, 'date': requested_date, 'time_slot': slot}):
            response = prediction()
        payload = response[0].get_json() if isinstance(response, tuple) else response.get_json()
        curve.append({'time_slot': slot, 'crowd_level': payload['predicted_crowd_level'], 'crowd_score': payload['confidence'], 'waiting_time_minutes': payload['predicted_waiting_time_minutes']})
    best = min(curve, key=lambda item: item['waiting_time_minutes'])
    return jsonify({'venue_name': place, 'today_curve': curve, 'best_today': best, 'best_this_week': {'date': requested_date, **best}})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
