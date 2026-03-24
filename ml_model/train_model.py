from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRAINING_DATA = {
    "hours_studied": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 4, 6, 8, 10, 2, 1, 3, 5, 7, 9, 6, 8, 10, 4, 5, 7, 3, 2, 9, 1, 2, 3, 5, 7, 8, 1, 4, 6, 9, 10, 2, 3, 4, 6, 7, 8, 9, 5, 2, 1, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 5, 6, 7, 8, 9, 4, 3, 2, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6],
    "attendance": [45, 50, 55, 60, 58, 65, 70, 68, 75, 80, 62, 66, 72, 78, 85, 48, 52, 57, 63, 69, 74, 76, 82, 88, 54, 46, 51, 59, 67, 73, 79, 84, 90, 61, 64, 71, 53, 49, 87, 44, 55, 58, 65, 72, 77, 49, 60, 68, 82, 88, 54, 59, 63, 70, 75, 81, 85, 66, 56, 50, 92, 84, 78, 72, 65, 60, 55, 50, 48, 45, 88, 82, 76, 70, 64, 58, 52, 48, 46, 95, 68, 72, 78, 84, 89, 62, 58, 54, 66, 71, 77, 83, 88, 94, 48, 53, 59, 64, 70, 75],
    "internal_marks": [10, 12, 15, 14, 18, 17, 20, 19, 22, 21, 13, 16, 18, 21, 24, 11, 14, 17, 19, 22, 20, 23, 25, 26, 15, 12, 16, 18, 21, 24, 22, 25, 27, 17, 19, 23, 14, 13, 26, 11, 14, 15, 18, 20, 22, 12, 16, 19, 24, 26, 13, 15, 17, 20, 21, 23, 25, 18, 14, 12, 28, 24, 22, 20, 18, 16, 14, 13, 12, 10, 25, 23, 21, 19, 17, 15, 13, 12, 11, 29, 19, 21, 23, 25, 27, 17, 15, 13, 19, 21, 23, 25, 27, 29, 11, 13, 16, 18, 20, 22],
    "cgpa": [5.4, 5.8, 6.1, 6.5, 6.0, 6.7, 7.0, 6.6, 7.2, 7.5, 6.2, 6.8, 7.1, 7.6, 8.0, 5.6, 6.0, 6.4, 6.9, 7.3, 7.0, 7.4, 7.8, 8.2, 6.1, 5.7, 6.2, 6.6, 7.0, 7.5, 7.3, 7.9, 8.4, 6.5, 6.7, 7.1, 6.0, 5.8, 8.1, 5.5, 5.9, 6.1, 6.6, 7.1, 7.5, 5.7, 6.3, 6.9, 7.8, 8.2, 5.9, 6.2, 6.5, 7.0, 7.3, 7.7, 8.0, 6.7, 6.0, 5.6, 8.8, 8.1, 7.6, 7.1, 6.6, 6.2, 5.9, 5.7, 5.5, 5.3, 8.4, 7.9, 7.4, 7.0, 6.5, 6.1, 5.8, 5.5, 5.4, 9.0, 6.9, 7.2, 7.6, 8.1, 8.5, 6.4, 6.1, 5.8, 6.8, 7.2, 7.6, 8.0, 8.4, 9.1, 5.5, 5.8, 6.2, 6.6, 7.0, 7.4],
    "last_sem_sgpa": [5.6, 6.0, 5.9, 6.7, 6.3, 6.5, 7.2, 6.4, 7.0, 7.6, 6.0, 6.6, 7.0, 7.4, 7.8, 5.9, 6.1, 6.3, 6.8, 7.1, 6.9, 7.3, 7.6, 8.0, 6.0, 5.8, 6.3, 6.5, 6.9, 7.4, 7.2, 7.7, 8.1, 6.6, 6.8, 7.0, 6.2, 5.9, 7.9, 5.7, 6.1, 6.3, 6.8, 7.3, 7.6, 5.8, 6.5, 7.1, 8.0, 8.4, 6.0, 6.4, 6.7, 7.2, 7.5, 7.9, 8.2, 6.9, 6.2, 5.8, 9.0, 8.3, 7.8, 7.3, 6.8, 6.4, 6.1, 5.9, 5.7, 5.5, 8.6, 8.1, 7.6, 7.2, 6.7, 6.3, 6.0, 5.7, 5.6, 9.2, 7.1, 7.4, 7.8, 8.3, 8.7, 6.6, 6.3, 6.0, 7.0, 7.4, 7.8, 8.2, 8.6, 9.3, 5.7, 6.0, 6.4, 6.8, 7.2, 7.6],
    "result": ["Fail", "Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Fail", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass"],
}


def train_model():
    X = pd.DataFrame(TRAINING_DATA).drop("result", axis=1)
    y = TRAINING_DATA["result"]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


if __name__ == "__main__":
    model = train_model()
    output_path = Path(__file__).resolve().parent / "student_model.joblib"
    joblib.dump(model, output_path)
    print(f"Saved model to {output_path}")
