import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv("transport_data.csv")

# Load models
time_model = pickle.load(open("time_model.pkl", "rb"))
crowd_model = pickle.load(open("crowd_model.pkl", "rb"))

# Load encoders
le_time = pickle.load(open("le_time.pkl", "rb"))
le_day = pickle.load(open("le_day.pkl", "rb"))
le_crowd = pickle.load(open("le_crowd.pkl", "rb"))

# Title
st.title("🚍 Public Transport Predictor")

st.write("Predict travel time and crowd level based on input")

# Inputs
distance = st.number_input("Enter distance (km)", min_value=0.0, step=0.5)

time = st.selectbox("Select time of day", ["morning", "afternoon", "evening", "night"])
day = st.selectbox("Select day type", ["weekday", "weekend"])

# Button
if st.button("Predict"):

    # Encode
    time_encoded = le_time.transform([time])[0]
    day_encoded = le_day.transform([day])[0]

    # DataFrame
    input_data = pd.DataFrame(
        [[distance, time_encoded, day_encoded]],
        columns=["distance_km", "time_of_day", "day_type"]
    )

    # Predict
    time_pred = time_model.predict(input_data)
    crowd_pred = crowd_model.predict(input_data)

    crowd_label = le_crowd.inverse_transform(crowd_pred)

    # Output
    st.subheader("Prediction Result")
    st.write(f"⏱ Travel Time: {round(time_pred[0])} minutes")
    st.write(f"👥 Crowd Level: {crowd_label[0]}")

    st.subheader("📊 Data Insights")

# Graph 1: Distance vs Travel Time
st.write("Distance vs Travel Time")

plt.figure()
plt.scatter(df["distance_km"], df["travel_time_min"])
plt.xlabel("Distance (km)")
plt.ylabel("Travel Time (min)")
st.pyplot(plt)

# Graph 2: Crowd Level Count
st.write("Crowd Level Distribution")

plt.figure()
df["crowd_level"].value_counts().plot(kind="bar")
plt.xlabel("Crowd Level")
plt.ylabel("Count")
st.pyplot(plt)

# Graph 3: Average Travel Time by Time of Day
st.write("Average Travel Time by Time of Day")

plt.figure()
df.groupby("time_of_day")["travel_time_min"].mean().plot(kind="bar")
plt.xlabel("Time of Day")
plt.ylabel("Avg Travel Time")
plt.tight_layout()
st.pyplot(plt)