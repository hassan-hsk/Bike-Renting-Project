
# 🚲 Bike Sharing Demand Prediction

This project aims to predict bike sharing demand using historical data. It combines a **Streamlit frontend**, a **FastAPI backend**, and multiple **machine learning models** trained and tracked via **MLflow**.

---

## 📁 Project Structure

```
├── .github/workflows/            # CI/CD pipelines (GitHub Actions)
├── .streamlit/config.toml        # Streamlit UI configuration
├── backend/                      # FastAPI backend
│   ├── main.py                   # API entry point
│   ├── model_loader.py           # Loads trained models
│   ├── schemas.py                # Pydantic models for API
│   ├── models/                   # (optional) model helpers
├── data/bike_sharing.csv         # Dataset
├── frontend/app.py               # Streamlit app UI
├── mlruns/                       # MLflow tracking artifacts
├── models/                       # Saved ML models and metrics
│   ├── decision_tree.pkl
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── r2_scores.json
├── venv/                         # Python virtual environment
├── __pycache__/                  # Compiled Python files
```

---

## 🚀 Features

- 🧠 Trains multiple models: Linear Regression, Decision Tree, Random Forest
- 📈 Tracks experiments using MLflow
- 🎯 Shows R² scores for evaluation
- 🖥️ Streamlit interface for predictions
- ⚙️ FastAPI backend for serving predictions

---

## 📊 Dataset

The project uses the **bike_sharing.csv** dataset, which contains:

- Date & time
- Weather conditions
- Temperature, humidity
- Count of bike rentals (target)

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/bike-sharing-prediction.git
   cd bike-sharing-prediction
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate (Windows)
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Running the App

1. **Start the backend**:
   ```bash
   uvicorn backend.main:app --reload
   ```

2. **Start the frontend**:
   ```bash
   streamlit run frontend/app.py
   ```

---

## 📦 Trained Models

All models are pre-trained and stored in the `models/` folder as `.pkl` files. These are loaded by the backend to serve predictions.

---

## 🧪 MLflow Tracking

- MLflow is used to log experiments and metrics.
- To launch the MLflow UI:
  ```bash
  mlflow ui
  ```

---

## 📈 Evaluation Metrics

Models are evaluated based on:

- R² Score (stored in `models/r2_scores.json`)
- MAE, RMSE (optional — add for improvements)

---

## 👨‍💻 Author

**Hassan Sarfraz**  
_Data Science | COMSATS University Islamabad_

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.
