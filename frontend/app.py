import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Bike Sharing ML App", layout="wide")

page = st.sidebar.selectbox("Select Page", ["Model Comparison", "Predict Bike Demand"])

if page == "Model Comparison":
    st.title("🚴‍♂️ Model Accuracy Comparison - Bike Sharing Demand")

    response = requests.get(f"{BACKEND_URL}/scores")
    scores = response.json()

    acc_scores = {k: round(v * 100, 2) for k, v in scores.items()}
    st.write(pd.DataFrame.from_dict(acc_scores, orient='index', columns=["Accuracy (%)"]))

    fig, ax = plt.subplots()
    ax.bar(acc_scores.keys(), acc_scores.values(), color=['mediumseagreen', 'coral', 'steelblue'])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim([0, 100])
    st.pyplot(fig)

elif page == "Predict Bike Demand":
    st.title("🔍 Predict Total Bike Rentals")

    col1, col2 = st.columns(2)

    with col1:
        season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
        season = ["Spring", "Summer", "Fall", "Winter"].index(season) + 1

        yr = st.selectbox("Year", [2011, 2012])
        yr = 0 if yr == 2011 else 1

        mnth = st.slider("Month", 1, 12, 6)
        day = st.slider("Day of Month", 1, 31, 15)

        holiday = st.selectbox("Holiday?", ["No", "Yes"])
        holiday = 1 if holiday == "Yes" else 0

    with col2:
        weekday = st.slider("Weekday (0=Sun ... 6=Sat)", 0, 6, 2)

        dayofweek = st.selectbox("Day of Week", [
            "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"
        ])
        dayofweek = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"].index(dayofweek)

        workingday = st.selectbox("Working Day?", ["Yes", "No"])
        workingday = 1 if workingday == "Yes" else 0

        weathersit = st.selectbox("Weather Situation", [
            "1 - Clear, Few clouds", "2 - Mist + Cloudy", "3 - Light Snow/Rain", "4 - Heavy Rain/Ice"
        ])
        weathersit = int(weathersit[0])

    st.markdown("### 🌡️ Weather Conditions (Real Values)")

    temp_real = st.slider("Temperature (°C)", -8.0, 39.0, 20.0)
    atemp_real = st.slider("Feels Like Temperature (°C)", -16.0, 50.0, 22.0)
    hum_real = st.slider("Humidity (%)", 0, 100, 60)
    windspeed_real = st.slider("Windspeed (km/h)", 0.0, 67.0, 15.0)

    temp = (temp_real + 8) / 47
    atemp = (atemp_real + 16) / 66
    hum = hum_real / 100
    windspeed = windspeed_real / 67

    model_choice = st.selectbox("Choose Model", ["Random Forest", "Decision Tree", "Linear Regression"])

    if st.button("Predict Demand"):
        payload = {
            "season": season,
            "yr": yr,
            "mnth": mnth,
            "holiday": holiday,
            "weekday": weekday,
            "workingday": workingday,
            "weathersit": weathersit,
            "temp": temp,
            "atemp": atemp,
            "hum": hum,
            "windspeed": windspeed,
            "day": day,
            "dayofweek": dayofweek,
            "model": model_choice
        }

        response = requests.post(f"{BACKEND_URL}/predict", json=payload)
        result = response.json()

        if "prediction" in result:
            st.success(f"🚲 Predicted Total Bike Rentals: **{result['prediction']} rentals**")
        else:
            st.error("❌ Prediction failed. Please try again.")
