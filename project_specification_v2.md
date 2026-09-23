# Project Specification v2.0 — AI-Based Smart Crowd Prediction and Visitor Assistance System

> **Agent build note:** This document is the single source of truth. Where a section is marked
> `⚠️ FILL IN`, the human must supply the value before the build starts. Do not invent values
> for those fields. Everything else is a binding contract — do not substitute alternatives.

---

## 0. Build Order (follow strictly, verify each checkpoint before moving on)

| Phase | Deliverable | Checkpoint (must pass before continuing) |
|---|---|---|
| 0 | Repo scaffold, `.env.example`, `requirements.txt`, `package.json` | `uvicorn` starts, Vite dev server starts, both return a health page |
| 1 | Dataset loader + schema validation script | `python ml_pipeline/validate_dataset.py` prints row count and passes all assertions in §3.3 |
| 2 | Training pipeline, serialized artifacts | `saved_models/` contains all 5 artifacts in §4.4; metrics printed match §4.5 targets |
| 3 | `ml_service.py` offline inference | `python -m backend.services.ml_service --selftest` returns a valid prediction with no API calls |
| 4 | `external_service.py` + mapping layer | Unit tests in §5.4 pass against recorded API fixtures |
| 5 | FastAPI routes + MongoDB | Every endpoint in §7 returns the documented shape; Swagger docs render |
| 6 | Frontend pages 1–4 | All pages render with live backend data |
| 7 | Admin auth + dashboard (page 5) | Login → JWT cookie → protected route works; unauthorized returns 401 |
| 8 | Polish, responsive pass, README | Acceptance criteria in §11 all checked |

**Rule:** the system must be fully functional with external APIs disabled (`USE_LIVE_APIS=false`).
Live data is an enhancement layer, never a hard dependency.

---

## 1. Project Overview

An AI/ML web platform that predicts crowd levels and estimated waiting times at public venues
(temples as the primary demonstration use case, with the framework generalizing to tourist
attractions, stadiums, and malls). It fuses a historical dataset with live weather and traffic
conditions, serves visitors a "best time to visit" recommendation, and gives venue administrators
an analytics dashboard.

**Scope boundary:** the model estimates the *most probable* crowd level from historical patterns
and current conditions. It does not claim to forecast the future with certainty. This framing must
appear in the UI (see §9.3).

---

## 2. Tech Stack (locked — no substitutions)

| Layer | Choice | Notes |
|---|---|---|
| Backend language | Python 3.10+ | |
| Backend framework | FastAPI + Uvicorn | async routes, Pydantic v2 |
| ML | scikit-learn, pandas, numpy, joblib | |
| Database | MongoDB via **Motor** (async driver) | not PyMongo sync |
| HTTP client | httpx (async) | |
| Cache | in-memory TTL cache (`cachetools.TTLCache`) | Redis explicitly out of scope for v1 |
| Frontend | **Vite + React 18 + React Router v6** | **NOT Next.js.** The original spec listed both; React Router is authoritative because the folder structure is `src/pages` + `App.jsx` |
| Styling | Tailwind CSS v3 | |
| Charts | Recharts | |
| Icons | Lucide React | |
| Animation | Framer Motion | |
| Auth | JWT in HTTP-only cookie, bcrypt via `passlib` | |

**Ports:** backend `8000`, frontend `5173`. CORS allowlist: `http://localhost:5173` plus
`FRONTEND_ORIGIN` from env.

---

## 3. Dataset Contract  ⚠️ CRITICAL

### 3.1 File location
`ml_pipeline/dataset/crowd_data.csv` — the human places the real file here before Phase 1.

### 3.2 Column schema

| Column | Type | Allowed values / range |
|---|---|---|
| `place_name` | categorical str | ⚠️ FILL IN — exact venue strings as they appear in your CSV |
| `date` | date `YYYY-MM-DD` | |
| `day_of_week` | categorical str | `Monday`…`Sunday` |
| `time_slot` | categorical str | see §3.4 |
| `is_weekend` | int | `0`, `1` |
| `is_public_holiday` | int | `0`, `1` |
| `is_festival` | int | `0`, `1` |
| `weather_condition` | categorical str | `Clear`, `Clouds`, `Rain`, `Thunderstorm`, `Mist`, `Haze` |
| `temperature` | float | °C, 5–50 |
| `traffic_level` | ordinal str | `Low`, `Medium`, `High`, `Severe` |
| `parking_occupancy` | float | 0–100 (percent) |
| `mobile_device_density` | float | 0–100 (normalized index) |
| `historical_avg_visitors` | int | ≥ 0 |
| `crowd_level` | **target (classifier)** ordinal str | `Low`, `Medium`, `High`, `Very High` |
| `waiting_time_minutes` | **target (regressor)** float | ≥ 0 |

> ⚠️ **FILL IN:** if your actual CSV uses different column names or category spellings,
> correct this table rather than renaming your data. If a listed column is absent from your CSV,
> delete the row here and remove it from §4.2 as well.

### 3.3 Validation assertions (`ml_pipeline/validate_dataset.py`)
The script must fail loudly if any of these break:
- every column in §3.2 is present, no extras
- zero nulls in target columns; other nulls reported with counts
- every categorical value is inside its allowed set (print offenders)
- `crowd_level` class counts printed; warn if the smallest class is < 5% of rows
- row count ≥ 500 (warn below 2000 — note limited-data risk in the README)

### 3.4 Time slot enumeration (fixed, 12 slots)
`05:00-07:00`, `07:00-09:00`, `09:00-11:00`, `11:00-13:00`, `13:00-15:00`, `15:00-17:00`,
`17:00-19:00`, `19:00-21:00`, `21:00-23:00`, `23:00-01:00`, `01:00-03:00`, `03:00-05:00`

A helper `time_slot_for(dt) -> str` must live in `backend/utils/slots.py` and be the **only**
place this mapping exists.

---

## 4. ML Pipeline Contract

### 4.1 Split
Stratified on `crowd_level`, `test_size=0.2`, `random_state=42`. Use the same split indices for
classifier and regressor so metrics are comparable.

### 4.2 Encoding (binding — training and inference must share this code)

Implemented once in `ml_pipeline/encoders.py`, imported by both `train.py` and `ml_service.py`.

- **Ordinal (explicit maps, never `LabelEncoder` on the fly):**
  - `traffic_level`: Low 0, Medium 1, High 2, Severe 3
  - `crowd_level` (target): Low 0, Medium 1, High 2, Very High 3
- **One-hot:** `place_name`, `day_of_week`, `time_slot`, `weather_condition`
  using `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
- **Passthrough numeric:** `is_weekend`, `is_public_holiday`, `is_festival`, `temperature`,
  `parking_occupancy`, `mobile_device_density`, `historical_avg_visitors`
- **No scaling.** Random Forests do not require it; adding a scaler creates a needless artifact.
- **Derived features** (computed identically at train and inference):
  - `month` (1–12) and `slot_index` (0–11) from `date` / `time_slot`

The fitted `ColumnTransformer` is serialized and reused. **Never re-fit an encoder at inference.**

### 4.3 Models
```python
RandomForestClassifier(random_state=42, class_weight="balanced")
RandomForestRegressor(random_state=42)
```
Tuned with `GridSearchCV(cv=5, scoring="f1_macro" | "neg_root_mean_squared_error")` over:
`n_estimators` [200, 400], `max_depth` [None, 12, 20], `min_samples_split` [2, 5, 10],
`min_samples_leaf` [1, 2, 4].

### 4.4 Artifacts written to `ml_pipeline/saved_models/`
1. `preprocessor.joblib` — fitted ColumnTransformer
2. `clf_crowd_level.joblib`
3. `reg_waiting_time.joblib`
4. `feature_names.json` — output feature order from the preprocessor
5. `metadata.json` — training date, row count, metrics, sklearn version, category sets

`ml_service.py` must assert the loaded sklearn version matches `metadata.json` and refuse to
serve if it differs.

### 4.5 Metrics to print and store
- Classifier: accuracy, macro F1 (**target > 0.88**), per-class precision/recall, confusion matrix
- Regressor: RMSE, MAE, R²
- Feature importances for both, top 15, saved into `metadata.json` for the admin dashboard

If macro F1 lands below 0.88, **do not fake it** — record the real number and note the likely
cause (class imbalance or small dataset) in the README.

### 4.6 Confidence
`clf.predict_proba(X).max()` → returned as `confidence` (0–1, rounded to 2dp). Anything below
0.55 makes the API set `"low_confidence": true`, and the UI shows a caution note.

---

## 5. External API Integration Contract

### 5.1 Weather — OpenWeatherMap
- **Today / now:** `GET /data/2.5/weather?lat&lon&units=metric&appid=`
- **Future date (≤ 5 days):** `GET /data/2.5/forecast` — pick the 3-hour entry nearest the
  requested slot's midpoint
- **Future date (> 5 days) or API failure:** fall back to the dataset's modal
  `weather_condition` and mean `temperature` for that venue + month + slot.
  Response must set `"weather_source": "historical_fallback"`.

**Mapping OWM `weather[0].main` → `weather_condition`:**

| OWM value | Feature value |
|---|---|
| Clear | `Clear` |
| Clouds | `Clouds` |
| Rain, Drizzle | `Rain` |
| Thunderstorm | `Thunderstorm` |
| Mist, Fog | `Mist` |
| Haze, Smoke, Dust, Sand | `Haze` |
| Snow, anything else | `Clouds` (documented default) |

`temperature` = `main.temp` (already °C with `units=metric`), clamped to 5–50.

### 5.2 Traffic — TomTom Flow Segment Data
`GET /traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key=`

Compute `ratio = currentSpeed / freeFlowSpeed`, then bucket:

| ratio | `traffic_level` |
|---|---|
| ≥ 0.85 | `Low` |
| 0.60 – 0.85 | `Medium` |
| 0.40 – 0.60 | `High` |
| < 0.40 | `Severe` |

**Traffic is only meaningful for "now."** For any future slot, use the dataset's modal
`traffic_level` for that venue + day_of_week + slot, and set `"traffic_source": "historical"`.

### 5.3 Features with no live source
`parking_occupancy` and `mobile_device_density` have **no public API**. At inference:
- default = dataset mean for that venue + day_of_week + time_slot
- if that combination is unseen, fall back to the venue mean, then the global mean
- both remain **user-overridable** via optional request fields, so the demo can show sensitivity

These lookup tables are precomputed during training and saved as
`saved_models/fallback_tables.json`.

### 5.4 Reliability rules
- Per-call timeout 4s, 1 retry with 500ms backoff
- `TTLCache(maxsize=256, ttl=600)` keyed on `(provider, round(lat,3), round(lon,3))`
- Any API failure is **non-fatal**: log a warning, use the fallback, flag the source in the response
- `USE_LIVE_APIS=false` skips all outbound calls entirely
- Missing API key at startup → log a warning and auto-disable live APIs (do not crash)

### 5.5 `.env.example` (must be committed; real `.env` git-ignored)
```
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=crowd_prediction
JWT_SECRET=change_me_to_a_long_random_string
JWT_EXPIRE_MINUTES=120
OPENWEATHER_API_KEY=
TOMTOM_API_KEY=
USE_LIVE_APIS=true
FRONTEND_ORIGIN=http://localhost:5173
ADMIN_BOOTSTRAP_EMAIL=admin@example.com
ADMIN_BOOTSTRAP_PASSWORD=ChangeThis123!
```

---

## 6. Database (MongoDB, via Motor)

### `users`
`_id`, `username`, `email` (unique index), `password_hash`, `role` (`admin`), `created_at`

### `venues`
`_id`, `venue_name` (unique index, **must exactly match `place_name` values in the CSV**),
`category` (`Temple` | `Tourist Attraction` | `Stadium` | `Mall` | `Event Venue`),
`latitude`, `longitude`, `max_capacity`, `operating_hours`, `description`, `image_url`

### `historical_records`
`venue_id`, `timestamp`, `day_of_week`, `time_slot`, `weather_condition`, `temperature`,
`traffic_level`, `parking_occupancy`, `mobile_device_density`, `actual_visitors`,
`actual_waiting_time`. Index: `(venue_id, timestamp)`

### `prediction_logs`
`_id`, `venue_name`, `input_features` (object), `predicted_crowd_level`,
`predicted_waiting_time`, `confidence`, `weather_source`, `traffic_source`, `timestamp`
Index: `timestamp` descending

### Seeding
`backend/seed.py` must:
1. create the bootstrap admin from env vars (idempotent — skip if the email exists)
2. insert one `venues` document per distinct `place_name` in the CSV
   ⚠️ **FILL IN** real lat/long, category, and operating hours for each venue — coordinates
   drive the live weather and traffic calls, so placeholder values will produce wrong data
3. bulk-load the CSV rows into `historical_records`

---

## 7. API Contract

All responses use these exact shapes. Errors use
`{"detail": "<human readable message>", "code": "<SCREAMING_SNAKE>"}`.

### `GET /api/health`
`{"status": "ok", "models_loaded": true, "live_apis": true}`

### `GET /api/venues?category=&q=`
```json
[{"id":"...","venue_name":"...","category":"Temple","latitude":17.36,"longitude":78.47,
  "operating_hours":"05:00-21:00","image_url":"...",
  "live_status":{"weather_condition":"Clear","temperature":31.2,"traffic_level":"Medium"}}]
```
`live_status` may be `null` if live APIs are off — the frontend must handle that.

### `POST /api/predict`
Request:
```json
{"venue_name":"...","date":"2026-09-26","time_slot":"09:00-11:00",
 "is_public_holiday":false,"is_festival":false,
 "overrides":{"parking_occupancy":null,"mobile_device_density":null,
              "weather_condition":null,"temperature":null,"traffic_level":null}}
```
Response:
```json
{"venue_name":"...","date":"2026-09-26","time_slot":"09:00-11:00",
 "predicted_crowd_level":"High","class_probabilities":{"Low":0.05,"Medium":0.18,"High":0.62,"Very High":0.15},
 "confidence":0.62,"low_confidence":false,
 "predicted_waiting_time_minutes":34.5,
 "resolved_features":{"weather_condition":"Clear","temperature":31.2,"traffic_level":"Medium",
                      "parking_occupancy":68.0,"mobile_device_density":72.4,
                      "is_weekend":true,"historical_avg_visitors":4200},
 "weather_source":"live","traffic_source":"historical",
 "top_factors":[{"feature":"time_slot=09:00-11:00","importance":0.21}],
 "disclaimer":"Estimated from historical patterns and current conditions; not a guaranteed forecast."}
```
`resolved_features` is required — it is what makes the system explainable, and the UI renders it.

Every call writes a `prediction_logs` document.

### `GET /api/recommendations/{venue_name}?date=&horizon_days=7`
Runs inference across all operating time slots for the requested date and the next
`horizon_days`, returns:
```json
{"venue_name":"...",
 "today_curve":[{"time_slot":"05:00-07:00","crowd_level":"Low","crowd_score":0.8,"waiting_time_minutes":6.2}],
 "best_today":{"time_slot":"07:00-09:00","waiting_time_minutes":9.1,"crowd_level":"Low"},
 "best_this_week":{"date":"2026-09-24","time_slot":"07:00-09:00","waiting_time_minutes":5.4}}
```
**Ranking rule:** sort ascending by `predicted_waiting_time_minutes`; break ties with the lower
ordinal `crowd_level`, then the earlier slot. Only include slots inside `operating_hours`.

### Auth
- `POST /api/auth/register` — **disabled in production** unless `ALLOW_REGISTRATION=true`; admins
  come from the bootstrap seed
- `POST /api/auth/login` → sets HTTP-only, SameSite=Lax cookie; body returns `{"username","role"}`
- `POST /api/auth/logout`
- `GET /api/auth/me` → 401 when unauthenticated

### Admin (all require the admin role)
- `GET /api/admin/analytics` → total predictions, predictions per day (last 30), crowd-level
  distribution, top venues by query volume, model metrics + feature importances from
  `metadata.json`, system health
- `POST /api/admin/venue` — create/update a venue

Rate limit: 30 requests/minute per IP on `/api/predict` and `/api/recommendations/*`.

---

## 8. Backend Structure

```
backend/
├── main.py                  # app, CORS, exception handlers, startup model load
├── config.py                # pydantic-settings, reads .env
├── database.py              # Motor client + index creation
├── seed.py
├── models/                  # Pydantic request/response schemas (mirror §7 exactly)
├── routers/                 # health, venues, predict, recommendations, auth, admin
├── services/
│   ├── ml_service.py        # load artifacts, build feature frame, predict, explain
│   ├── external_service.py  # OWM + TomTom clients, mapping, cache, fallbacks
│   └── auth_service.py      # bcrypt, JWT issue/verify, cookie handling
└── utils/slots.py           # single source of truth for time-slot logic
```

Models load **once** at startup into app state. A request must never read from disk.

---

## 9. Frontend

### 9.1 Routes (React Router v6)
`/` Home · `/venues` Directory · `/predict` Prediction Portal ·
`/recommendations` Recommendation Finder · `/admin/login` · `/admin/dashboard` (protected)

### 9.2 Page requirements
- **Home:** hero with venue search (navigates to `/predict?venue=`), live counters sourced from
  `/api/admin/analytics` public subset (never fabricated numbers), feature grid, how-it-works
  strip showing the data → model → recommendation flow.
- **Venues:** responsive card grid, category filter pills, search box, live status badges,
  skeleton loaders while fetching, empty state when filters match nothing.
- **Predict:** split layout — form on the left (venue, date picker, slot selector, holiday and
  festival toggles, an "advanced" disclosure for the optional overrides), results on the right
  (animated crowd-level badge, wait-time counter that counts up, probability bar for all four
  classes, a `resolved_features` panel labelled with `live` vs `historical` chips, and the
  `top_factors` list).
- **Recommendations:** Recharts area/line chart of the day's crowd curve with the best window
  highlighted, best-today and best-this-week pills, one-click jump back into `/predict`.
- **Admin:** login form → dashboard with prediction-volume line chart, crowd-level distribution
  bar chart, feature-importance horizontal bar chart, recent `prediction_logs` table, model
  metrics card.

### 9.3 Required UI honesty elements
- The disclaimer string from `/api/predict` renders visibly with every prediction.
- `weather_source` / `traffic_source` are shown as chips (`Live` green / `Historical` amber).
- When `low_confidence` is true, show a caution banner rather than hiding the uncertainty.

### 9.4 Design language
Deep indigo/slate dark base; emerald → amber → orange → red for Low → Medium → High → Very High
(use this exact ramp everywhere, including charts). Glassmorphism cards with subtle gradient
borders. Framer Motion for page transitions and card hovers. Fully responsive — mobile-first for
visitor pages, desktop-optimized for admin. Meet WCAG AA contrast; never rely on color alone for
crowd level (always pair with the text label).

---

## 10. Security
bcrypt password hashing · JWT in HTTP-only SameSite cookies · CORS restricted to the allowlist ·
Pydantic validation on every input · rate limiting per §7 · secrets only from env, never
committed · no PII stored; `mobile_device_density` is an anonymous aggregate index only, and the
README must state this explicitly.

---

## 11. Acceptance Criteria

- [ ] `validate_dataset.py` passes against the real CSV
- [ ] `train.py` produces all 5 artifacts and prints real metrics
- [ ] Classifier macro F1 reported honestly (target > 0.88)
- [ ] Prediction works end-to-end with `USE_LIVE_APIS=false`
- [ ] Prediction works with live APIs and correctly labels the data source
- [ ] Killing network access mid-session degrades to fallbacks without a 500
- [ ] A future date > 5 days out still returns a prediction
- [ ] Recommendations return a best slot consistent with the ranking rule
- [ ] Unauthenticated `/api/admin/*` returns 401; admin login works
- [ ] All five pages responsive at 375px and 1440px
- [ ] `prediction_logs` grows with each prediction and appears in the dashboard
- [ ] README documents setup, env vars, training, seeding, and honest model limitations

---

## 12. Known Limitations (state these in the README — do not paper over them)
1. Trained on a simulated/limited dataset; accuracy on real crowds is unvalidated.
2. Live traffic applies only to "now"; future slots use historical patterns.
3. Parking occupancy and device density have no live feed in v1.
4. Weather forecasts are reliable only ~5 days out.
5. The model estimates probability, not certainty.
