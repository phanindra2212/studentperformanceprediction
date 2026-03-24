import os
import joblib
from django.shortcuts import render
from django.conf import settings

# --- 1. MODEL LOADING CONFIGURATION ---
# This looks for the 'ml_model' folder in your project root
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_model', 'student_model.joblib')

# We load the model once when the server starts to save memory and time
if os.path.exists(MODEL_PATH):
    try:
        MODEL = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        MODEL = None
else:
    MODEL = None
    print(f"Warning: Model file not found at {MODEL_PATH}")


# --- 2. THE HOME VIEW ---
def home(request):
    """
    Handles the student data input form and returns the Pass/Fail prediction.
    """
    # Define the exact features the model expects
    feature_names = [
        "hours_studied", 
        "attendance", 
        "internal_marks", 
        "cgpa", 
        "last_sem_sgpa"
    ]

    context = {
        "prediction": None,
        "error": None,
        "form_data": {},  # Keeps user input in the fields after they click submit
    }

    if request.method == "POST":
        # Collect data from the POST request
        context["form_data"] = {
            field: request.POST.get(field, "").strip() for field in feature_names
        }

        try:
            # 1. Check if the model exists
            if MODEL is None:
                raise Exception("The machine learning model is not loaded. Please check the 'ml_model' folder.")

            # 2. Convert string inputs to floats
            # This will trigger a ValueError if a field is empty or contains letters
            input_values = [
                float(context["form_data"][field]) for field in feature_names
            ]

            # 3. Predict using the Random Forest model
            # [input_values] wraps the list in another list because scikit-learn expects 2D data
            result = MODEL.predict([input_values])[0]
            
            # 4. Store the result in context
            context["prediction"] = result

        except ValueError:
            context["error"] = "Invalid input! Please ensure all fields contain only numbers."
        except Exception as e:
            context["error"] = str(e)

    return render(request, "home.html", context)
