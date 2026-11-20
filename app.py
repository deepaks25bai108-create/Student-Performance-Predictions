from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load trained ML model
model = joblib.load("Models/student_model.pkl")


@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running"
    }


@app.post("/predict")
def predict(study_hours: float, attendance: float, previous_marks: float):

    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks
    }])

    prediction = model.predict(input_data)

    return {
        "prediction_result": str(prediction[0])
    }
