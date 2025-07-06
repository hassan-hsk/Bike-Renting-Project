# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath="bike_sharing.csv", scale_data=True, test_size=0.2, random_state=42):
    df = pd.read_csv(filepath)

    # --- Basic cleaning ---
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # --- Date formatting ---
    df['dteday'] = pd.to_datetime(df['dteday'])

    # Optional feature engineering from date
    df['day'] = df['dteday'].dt.day
    df['dayofweek'] = df['dteday'].dt.dayofweek

    df['temp'] = df['temp'].clip(0, 1)
    df['atemp'] = df['atemp'].clip(0, 1)
    df['hum'] = df['hum'].clip(0, 1)
    df['windspeed'] = df['windspeed'].clip(0, 1)

    # --- Feature selection ---
 
    df.drop(columns=['instant', 'dteday', 'casual', 'registered'], inplace=True)

    # Define features and target
    features = df.drop(columns=['cnt'])
    target = df['cnt']

    # --- Scaling numeric features (optional) ---
    if scale_data:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        X = pd.DataFrame(features_scaled, columns=features.columns)
    else:
        X = features

    y = target

    # --- Optional split for training/testing ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
