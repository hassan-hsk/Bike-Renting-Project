import joblib

model_scores = {
    "Random Forest": joblib.load("backend/models/random_forest.pkl"),
    "Decision Tree": joblib.load("backend/models/decision_tree.pkl"),
    "Linear Regression": joblib.load("backend/models/linear_regression.pkl")
}
