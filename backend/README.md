# AI-Based Smart Crowd Prediction and Visitor Assistance System

This Flask service trains Random Forest models from the supplied 36,500-row CSV. The source schema is preserved; `train_model.py` derives canonical time slots and numeric features from the provided columns. Mobile density is an anonymous aggregate category, never personal data.

Run:

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```
