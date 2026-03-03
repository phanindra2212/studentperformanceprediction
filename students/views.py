from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Base training data
TRAINING_DATA = {
    "hours_studied": [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        2, 4, 6, 8, 10, 1, 3, 5, 7, 9,
        4, 6, 8, 10, 2, 1, 3, 5, 7, 9,
        6, 8, 10, 4, 5, 7, 3, 2, 9, 1,
    ],
    "attendance": [
        45, 50, 55, 60, 58, 65, 70, 68, 75, 80,
        62, 66, 72, 78, 85, 48, 52, 57, 63, 69,
        74, 76, 82, 88, 54, 46, 51, 59, 67, 73,
        79, 84, 90, 61, 64, 71, 53, 49, 87, 44,
    ],
    "internal_marks": [
        10, 12, 15, 14, 18, 17, 20, 19, 22, 21,
        13, 16, 18, 21, 24, 11, 14, 17, 19, 22,
        20, 23, 25, 26, 15, 12, 16, 18, 21, 24,
        22, 25, 27, 17, 19, 23, 14, 13, 26, 11,
    ],
    "cgpa": [
        5.4, 5.8, 6.1, 6.5, 6.0, 6.7, 7.0, 6.6, 7.2, 7.5,
        6.2, 6.8, 7.1, 7.6, 8.0, 5.6, 6.0, 6.4, 6.9, 7.3,
        7.0, 7.4, 7.8, 8.2, 6.1, 5.7, 6.2, 6.6, 7.0, 7.5,
        7.3, 7.9, 8.4, 6.5, 6.7, 7.1, 6.0, 5.8, 8.1, 5.5,
    ],
    "last_sem_sgpa": [
        5.6, 6.0, 5.9, 6.7, 6.3, 6.5, 7.2, 6.4, 7.0, 7.6,
        6.0, 6.6, 7.0, 7.4, 7.8, 5.9, 6.1, 6.3, 6.8, 7.1,
        6.9, 7.3, 7.6, 8.0, 6.0, 5.8, 6.3, 6.5, 6.9, 7.4,
        7.2, 7.7, 8.1, 6.6, 6.8, 7.0, 6.2, 5.9, 7.9, 5.7,
    ],
    "result": [
        "Fail", "Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass",
        "Fail", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Fail",
    ],
}


FEATURE_COLUMNS = [
    "hours_studied",
    "attendance",
    "internal_marks",
    "cgpa",
    "last_sem_sgpa",
]


def build_model():
    data = pd.DataFrame(TRAINING_DATA)
    x = data[FEATURE_COLUMNS]
    y = data["result"]

    x_train, _, y_train, _ = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ]
    )
    model.fit(x_train, y_train)
    return model


MODEL = build_model()


def home(request):
    context = {
        "prediction": None,
        "error": None,
        "form_data": {},
    }

    if request.method == "POST":
        context["form_data"] = {
            key: request.POST.get(key, "").strip() for key in FEATURE_COLUMNS
        }

        try:
            input_row = [
                float(context["form_data"]["hours_studied"]),
                float(context["form_data"]["attendance"]),
                float(context["form_data"]["internal_marks"]),
                float(context["form_data"]["cgpa"]),
                float(context["form_data"]["last_sem_sgpa"]),
            ]

            prediction = MODEL.predict([input_row])[0]
            context["prediction"] = prediction
        except ValueError:
            context["error"] = "Please enter valid numeric values in all fields."
        except Exception as exc:
            context["error"] = f"Prediction failed: {exc}"

    return render(request, "home.html", context)
