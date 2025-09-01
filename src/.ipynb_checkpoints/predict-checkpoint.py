import pandas as pd
import joblib

model = joblib.load("../models/churn_model.joblib")

new_data = pd.DataFrame([
    {"feature1": 20, "feature2": 1, "feature3": 50},
    {"feature1": 45, "feature2": 0, "feature3": 10}
])

predictions = model.predict(new_data)
print("Предсказания:", predictions)
