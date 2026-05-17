from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


TRAINING_DATA = {
    "hours_studied": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 4, 6, 8, 10, 2, 1, 3, 5, 7, 9, 6, 8, 10, 4, 5, 7, 3, 2, 9, 1, 2, 3, 5, 7, 8, 1, 4, 6, 9, 10, 2, 3, 4, 6, 7, 8, 9, 5, 2, 1, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 5, 6, 7, 8, 9, 4, 3, 2, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6],
    "attendance": [45, 50, 55, 60, 58, 65, 70, 68, 75, 80, 62, 66, 72, 78, 85, 48, 52, 57, 63, 69, 74, 76, 82, 88, 54, 46, 51, 59, 67, 73, 79, 84, 90, 61, 64, 71, 53, 49, 87, 44, 55, 58, 65, 72, 77, 49, 60, 68, 82, 88, 54, 59, 63, 70, 75, 81, 85, 66, 56, 50, 92, 84, 78, 72, 65, 60, 55, 50, 48, 45, 88, 82, 76, 70, 64, 58, 52, 48, 46, 95, 68, 72, 78, 84, 89, 62, 58, 54, 66, 71, 77, 83, 88, 94, 48, 53, 59, 64, 70, 75],
    "internal_marks": [10, 12, 15, 14, 18, 17, 20, 19, 22, 21, 13, 16, 18, 21, 24, 11, 14, 17, 19, 22, 20, 23, 25, 26, 15, 12, 16, 18, 21, 24, 22, 25, 27, 17, 19, 23, 14, 13, 26, 11, 14, 15, 18, 20, 22, 12, 16, 19, 24, 26, 13, 15, 17, 20, 21, 23, 25, 18, 14, 12, 28, 24, 22, 20, 18, 16, 14, 13, 12, 10, 25, 23, 21, 19, 17, 15, 13, 12, 11, 29, 19, 21, 23, 25, 27, 17, 15, 13, 19, 21, 23, 25, 27, 29, 11, 13, 16, 18, 20, 22],
    "cgpa": [5.4, 5.8, 6.1, 6.5, 6.0, 6.7, 7.0, 6.6, 7.2, 7.5, 6.2, 6.8, 7.1, 7.6, 8.0, 5.6, 6.0, 6.4, 6.9, 7.3, 7.0, 7.4, 7.8, 8.2, 6.1, 5.7, 6.2, 6.6, 7.0, 7.5, 7.3, 7.9, 8.4, 6.5, 6.7, 7.1, 6.0, 5.8, 8.1, 5.5, 5.9, 6.1, 6.6, 7.1, 7.5, 5.7, 6.3, 6.9, 7.8, 8.2, 5.9, 6.2, 6.5, 7.0, 7.3, 7.7, 8.0, 6.7, 6.0, 5.6, 8.8, 8.1, 7.6, 7.1, 6.6, 6.2, 5.9, 5.7, 5.5, 5.3, 8.4, 7.9, 7.4, 7.0, 6.5, 6.1, 5.8, 5.5, 5.4, 9.0, 6.9, 7.2, 7.6, 8.1, 8.5, 6.4, 6.1, 5.8, 6.8, 7.2, 7.6, 8.0, 8.4, 9.1, 5.5, 5.8, 6.2, 6.6, 7.0, 7.4],
    "last_sem_sgpa": [5.6, 6.0, 5.9, 6.7, 6.3, 6.5, 7.2, 6.4, 7.0, 7.6, 6.0, 6.6, 7.0, 7.4, 7.8, 5.9, 6.1, 6.3, 6.8, 7.1, 6.9, 7.3, 7.6, 8.0, 6.0, 5.8, 6.3, 6.5, 6.9, 7.4, 7.2, 7.7, 8.1, 6.6, 6.8, 7.0, 6.2, 5.9, 7.9, 5.7, 6.1, 6.3, 6.8, 7.3, 7.6, 5.8, 6.5, 7.1, 8.0, 8.4, 6.0, 6.4, 6.7, 7.2, 7.5, 7.9, 8.2, 6.9, 6.2, 5.8, 9.0, 8.3, 7.8, 7.3, 6.8, 6.4, 6.1, 5.9, 5.7, 5.5, 8.6, 8.1, 7.6, 7.2, 6.7, 6.3, 6.0, 5.7, 5.6, 9.2, 7.1, 7.4, 7.8, 8.3, 8.7, 6.6, 6.3, 6.0, 7.0, 7.4, 7.8, 8.2, 8.6, 9.3, 5.7, 6.0, 6.4, 6.8, 7.2, 7.6],
    "result": ["Fail", "Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Fail", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass"],
    "backlogs": [3, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 4, 3, 2, 0, 0, 0, 3, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 2, 3, 3, 4, 0, 0, 0, 0, 0, 2, 3, 3, 4, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0],
    "assignment_submission": [50, 55, 60, 65, 75, 80, 85, 82, 90, 95, 60, 78, 82, 88, 95, 52, 58, 70, 78, 85, 82, 88, 92, 98, 62, 50, 72, 78, 85, 90, 88, 94, 98, 78, 82, 88, 58, 52, 95, 45, 55, 60, 76, 84, 88, 52, 76, 84, 92, 96, 60, 62, 74, 82, 86, 90, 94, 80, 62, 55, 98, 92, 86, 82, 78, 74, 60, 54, 50, 46, 94, 90, 84, 80, 75, 65, 58, 52, 48, 98, 80, 84, 88, 92, 96, 74, 68, 60, 80, 85, 90, 94, 96, 98, 54, 62, 72, 78, 84, 88],
    "lab_marks": [11, 12, 13, 14, 16, 17, 19, 18, 22, 23, 13, 17, 19, 21, 24, 11, 13, 16, 18, 21, 19, 22, 24, 25, 14, 11, 16, 18, 21, 23, 21, 24, 26, 17, 19, 22, 13, 12, 25, 10, 12, 13, 17, 20, 22, 11, 17, 20, 24, 25, 13, 14, 17, 20, 21, 23, 25, 18, 13, 12, 27, 24, 22, 20, 18, 17, 13, 12, 11, 10, 24, 23, 21, 19, 18, 15, 13, 12, 11, 28, 19, 21, 23, 24, 26, 17, 15, 13, 19, 21, 23, 24, 26, 28, 11, 13, 16, 18, 19, 21],
    "practice_hours": [1, 1, 2, 2, 4, 5, 6, 5, 8, 9, 2, 4, 6, 7, 10, 1, 3, 4, 6, 8, 5, 7, 8, 10, 2, 1, 3, 5, 6, 8, 7, 9, 10, 4, 5, 7, 2, 1, 9, 0, 1, 2, 4, 6, 8, 1, 4, 6, 9, 10, 2, 2, 3, 6, 7, 8, 9, 5, 1, 1, 10, 9, 7, 6, 5, 4, 2, 1, 1, 0, 9, 8, 7, 6, 5, 3, 2, 1, 1, 10, 5, 6, 7, 8, 9, 4, 2, 1, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 4, 6],
    "risk_factor": ["High", "High", "High", "High", "Low", "Low", "Low", "Low", "Low", "Low", "Medium", "Low", "Low", "Low", "Low", "High", "High", "Low", "Low", "Low", "Low", "Low", "Low", "Low", "Medium", "High", "Low", "Low", "Low", "Low", "Low", "Low", "Low", "Low", "Low", "Low", "Medium", "High", "Low", "High", "High", "High", "Low", "Low", "Low", "High", "Low", "Low", "Low", "Low", "Medium", "Medium", "Low", "Low", "Low", "Low", "Low", "Low", "Medium", "High", "Low", "Low", "Low", "Low", "Low", "Low", "Medium", "High", "High", "High", "Low", "Low", "Low", "Low", "Low", "Medium", "High", "High", "High", "Low", "Low", "Low", "Low", "Low", "Low", "Low", "Medium", "High", "Low", "Low", "Low", "Low", "Low", "Low", "High", "High", "Low", "Low", "Low", "Low"],
    "predicted_cgpa": [5.5, 5.7, 6.0, 6.3, 6.2, 6.8, 7.1, 6.8, 7.4, 7.6, 6.3, 6.9, 7.2, 7.7, 8.2, 5.7, 6.1, 6.5, 7.0, 7.4, 7.1, 7.5, 7.9, 8.4, 6.2, 5.6, 6.4, 6.7, 7.1, 7.6, 7.4, 8.0, 8.5, 6.6, 6.8, 7.2, 6.1, 5.7, 8.3, 5.3, 5.8, 6.0, 6.8, 7.3, 7.6, 5.6, 6.5, 7.0, 8.0, 8.4, 6.0, 6.1, 6.6, 7.1, 7.4, 7.8, 8.1, 6.8, 5.9, 5.5, 9.0, 8.3, 7.8, 7.3, 6.8, 6.4, 5.8, 5.6, 5.4, 5.1, 8.6, 8.1, 7.6, 7.1, 6.7, 6.0, 5.6, 5.4, 5.2, 9.2, 7.0, 7.3, 7.7, 8.2, 8.6, 6.5, 6.0, 5.6, 6.9, 7.3, 7.7, 8.1, 8.5, 9.2, 5.4, 5.9, 6.3, 6.7, 7.1, 7.5]
}


FEATURE_NAMES = [
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


def build_models():
    clf_result = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ]
    )
    clf_risk = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ]
    )
    reg_cgpa = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)),
        ]
    )
    return clf_result, clf_risk, reg_cgpa


def train_model():
    data = pd.DataFrame(TRAINING_DATA)
    X = data[FEATURE_NAMES]

    le_result = LabelEncoder()
    le_risk = LabelEncoder()
    y_result = le_result.fit_transform(data["result"])
    y_risk = le_risk.fit_transform(data["risk_factor"])
    y_cgpa = data["predicted_cgpa"]

    split_data = train_test_split(
        X,
        y_result,
        y_risk,
        y_cgpa,
        test_size=0.2,
        random_state=42,
    )
    (
        X_train,
        X_test,
        y_result_train,
        y_result_test,
        y_risk_train,
        y_risk_test,
        y_cgpa_train,
        y_cgpa_test,
    ) = split_data

    eval_clf_result, eval_clf_risk, eval_reg_cgpa = build_models()
    eval_clf_result.fit(X_train, y_result_train)
    eval_clf_risk.fit(X_train, y_risk_train)
    eval_reg_cgpa.fit(X_train, y_cgpa_train)

    result_train_predictions = eval_clf_result.predict(X_train)
    result_predictions = eval_clf_result.predict(X_test)
    risk_train_predictions = eval_clf_risk.predict(X_train)
    risk_predictions = eval_clf_risk.predict(X_test)
    cgpa_train_predictions = eval_reg_cgpa.predict(X_train)
    cgpa_predictions = eval_reg_cgpa.predict(X_test)

    result_train_accuracy = accuracy_score(y_result_train, result_train_predictions)
    result_test_accuracy = accuracy_score(y_result_test, result_predictions)
    risk_train_accuracy = accuracy_score(y_risk_train, risk_train_predictions)
    risk_test_accuracy = accuracy_score(y_risk_test, risk_predictions)
    cgpa_train_r2 = r2_score(y_cgpa_train, cgpa_train_predictions)
    cgpa_test_r2 = r2_score(y_cgpa_test, cgpa_predictions)

    result_gap = result_train_accuracy - result_test_accuracy
    risk_gap = risk_train_accuracy - risk_test_accuracy
    cgpa_r2_gap = cgpa_train_r2 - cgpa_test_r2
    is_overfitting = result_gap > 0.10 or risk_gap > 0.10 or cgpa_r2_gap > 0.15

    overfitting_report = {
        "test_samples": len(X_test),
        "result_train_accuracy": round(result_train_accuracy * 100, 2),
        "result_test_accuracy": round(result_test_accuracy * 100, 2),
        "result_accuracy_gap": round(result_gap * 100, 2),
        "academic_risk_train_accuracy": round(risk_train_accuracy * 100, 2),
        "academic_risk_test_accuracy": round(risk_test_accuracy * 100, 2),
        "academic_risk_accuracy_gap": round(risk_gap * 100, 2),
        "cgpa_train_r2": round(cgpa_train_r2, 4),
        "cgpa_test_r2": round(cgpa_test_r2, 4),
        "cgpa_r2_gap": round(cgpa_r2_gap, 4),
        "cgpa_test_mse": round(mean_squared_error(y_cgpa_test, cgpa_predictions), 4),
        "cgpa_test_mae": round(mean_absolute_error(y_cgpa_test, cgpa_predictions), 4),
        "is_overfitting": is_overfitting,
        "verdict": "Possible overfitting" if is_overfitting else "No strong overfitting signal",
    }

    clf_result, clf_risk, reg_cgpa = build_models()
    clf_result.fit(X, y_result)
    clf_risk.fit(X, y_risk)
    reg_cgpa.fit(X, y_cgpa)

    return {
        "clf_result": clf_result,
        "clf_risk": clf_risk,
        "reg_cgpa": reg_cgpa,
        "le_result": le_result,
        "le_risk": le_risk,
        "features": FEATURE_NAMES,
        "overfitting_report": overfitting_report,
    }


if __name__ == "__main__":
    model = train_model()
    output_path = Path(__file__).resolve().parent / "student_model.joblib"
    joblib.dump(model, output_path)
    print(f"Saved model to {output_path}")
    print("Overfitting report:")
    for key, value in model["overfitting_report"].items():
        print(f"{key}: {value}")
