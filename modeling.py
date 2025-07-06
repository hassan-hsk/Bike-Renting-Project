import mlflow
import mlflow.sklearn
from preprocessing import load_and_preprocess_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json

def train_models():
    # Load preprocessed and already split data
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/bike_sharing.csv")

    models = {
        "random_forest": RandomForestRegressor(random_state=42),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "linear_regression": LinearRegression()
    }

    r2_scores = {}

    os.makedirs("models", exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Bike Sharing Demand Prediction")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            # ✅ Save R² score to dictionary
            r2_scores[name] = r2

            mlflow.log_param("model_name", name)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

            model_path = f"models/{name}.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)

    # ✅ Write all R² scores to file
    with open("models/r2_scores.json", "w") as f:
        json.dump(r2_scores, f)

    print("✅ Training complete. Models and metrics logged to MLflow.")

if __name__ == "__main__":
    train_models()
