import os
import joblib
import pandas as pd
from django.shortcuts import render
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, "ml_model", "student_model.joblib")
DEFAULT_FEATURE_NAMES = [
    "hours_studied",
    "attendance",
    "internal_marks",
    "cgpa",
    "last_sem_sgpa",
    "backlogs",
    "assignment_submission",
    "lab_marks",
    "practice_hours",
]

if os.path.exists(MODEL_PATH):
    try:
        MODEL = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        MODEL = None
else:
    MODEL = None
    print(f"Warning: Model file not found at {MODEL_PATH}")


def get_feature_names():
    if isinstance(MODEL, dict):
        return MODEL.get("features", DEFAULT_FEATURE_NAMES)
    return DEFAULT_FEATURE_NAMES


def validate_model():
    required_keys = {"clf_result", "clf_risk", "reg_cgpa", "le_result", "le_risk"}
    if MODEL is None:
        raise Exception(
            "The machine learning model is not loaded. Please train the model first."
        )
    if not isinstance(MODEL, dict) or not required_keys.issubset(MODEL):
        raise Exception(
            "The machine learning model has an unexpected format. Please retrain it."
        )


def home(request):
    """
    Handles the student data input form and returns the Pass/Fail prediction.
    """
    feature_names = get_feature_names()

    context = {
        "prediction": None,
        "error": None,
        "form_data": {},
    }

    if request.method == "POST":
        context["form_data"] = {
            field: request.POST.get(field, "").strip() for field in feature_names
        }

        try:
            validate_model()

            input_values = [
                float(context["form_data"][field]) for field in feature_names
            ]
            x_input = pd.DataFrame([input_values], columns=feature_names)

            result_enc = MODEL["clf_result"].predict(x_input)[0]
            result = MODEL["le_result"].inverse_transform([int(result_enc)])[0]

            risk_enc = MODEL["clf_risk"].predict(x_input)[0]
            risk = MODEL["le_risk"].inverse_transform([int(risk_enc)])[0]

            predicted_cgpa = float(MODEL["reg_cgpa"].predict(x_input)[0])

            context["prediction"] = {
                "result": result,
                "academic_risk_level": risk,
                "predicted_cgpa": round(predicted_cgpa, 2),
            }

        except ValueError:
            context["error"] = "Invalid input! Please ensure all fields contain only numbers."
        except Exception as e:
            context["error"] = str(e)

    return render(request, "home.html", context)
