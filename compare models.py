import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from preprocessing import load_and_preprocess_data

def compare_models():
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/bike_sharing.csv")
    
    # Combine training data for correlation and distribution plots
    df_full = pd.concat([X_train, y_train], axis=1)

    # ========== BEFORE TRAINING PLOTS ==========
    os.makedirs("models/plots", exist_ok=True)
    print("📊 Plotting pre-model insights...")

    # 1. Target Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df_full['cnt'], kde=True, bins=30)
    plt.title('Distribution of Target Variable (cnt)')
    plt.xlabel('Bike Rentals')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("models/plots/target_distribution.png")
    plt.close()

    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_full.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation with cnt")
    plt.tight_layout()
    plt.savefig("models/plots/correlation_heatmap.png")
    plt.close()

    # ========== MODEL COMPARISON ==========
    model_names = ["random_forest", "decision_tree", "linear_regression"]
    metrics = {}

    for name in model_names:
        model_path = f"models/{name}.pkl"
        if not os.path.exists(model_path):
            print(f"❌ Model {name} not found. Skipping.")
            continue

        model = joblib.load(model_path)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        metrics[name] = {
            "R2 Score": r2,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae
        }

        # ========== AFTER TRAINING PLOTS PER MODEL ==========
        print(f"📉 Plotting results for {name}...")

        # Prediction vs Actual
        plt.figure(figsize=(8, 5))
        plt.scatter(y_test, preds, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Actual vs Predicted - {name}")
        plt.tight_layout()
        plt.savefig(f"models/plots/{name}_actual_vs_predicted.png")
        plt.close()

        # Residual Plot
        residuals = y_test - preds
        plt.figure(figsize=(8, 5))
        sns.histplot(residuals, kde=True, bins=30)
        plt.title(f"Residuals Distribution - {name}")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"models/plots/{name}_residuals.png")
        plt.close()

    # ========== COMPARISON BAR PLOTS ==========
    print("📊 Creating comparison plots...")

    metrics_df = pd.DataFrame(metrics).T

    for metric in ['R2 Score', 'RMSE', 'MAE']:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=metrics_df.index, y=metrics_df[metric], palette='Blues_d')
        plt.title(f"{metric} Comparison")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.tight_layout()
        plt.savefig(f"models/plots/compare_{metric.lower().replace(' ', '_')}.png")
        plt.close()

    # Save metrics to JSON
    with open("models/metrics_comparison.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Comparison complete. Metrics and plots saved in 'models/plots/'")

if __name__ == "__main__":
    compare_models()
