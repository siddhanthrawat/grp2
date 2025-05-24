import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("🧠 Pandemica - Disease Outbreak Predictor")
st.subheader("AI-driven early warning system for infectious disease surveillance")

st.sidebar.header("Simulate or Upload Data")
data_mode = st.sidebar.radio("Choose Input Mode:", ("Simulate Synthetic Data", "Upload CSV"))

# Define symptoms
symptoms = [
    'fever_cases',
    'rash_cases',
    'platelet_alerts',
    'malnutrition_cases',
    'conjunctivitis_cases',
    'jaundice_cases'
]

# Load or simulate data
if data_mode == "Simulate Synthetic Data":
    days = np.arange(30)
    data = pd.DataFrame({
        'day': days,
        'fever_cases': np.random.poisson(lam=50, size=30),
        'rash_cases': np.random.poisson(lam=5, size=30),
        'platelet_alerts': np.random.poisson(lam=2, size=30),
        'malnutrition_cases': np.random.poisson(lam=3, size=30),
        'conjunctivitis_cases': np.random.poisson(lam=4, size=30),
        'jaundice_cases': np.random.poisson(lam=2, size=30),
    })
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        missing_cols = [col for col in ['day'] + symptoms if col not in data.columns]
        if missing_cols:
            st.error(f"Missing columns in uploaded file: {', '.join(missing_cols)}")
            st.stop()
    else:
        st.warning("Please upload a CSV file with columns: 'day' + symptom columns.")
        st.stop()

# Show the actual data
st.write("### Clinical Data Preview")
st.dataframe(data.head())

# Normalize
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[symptoms])

# Prepare sequences
X, y = [], []
window_size = 5
for i in range(len(scaled_features) - window_size):
    X.append(scaled_features[i:i + window_size])
    y.append(scaled_features[i + window_size])  # Predict all symptoms

X, y = np.array(X), np.array(y)

# LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(len(symptoms)))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

# Predict next day
last_sequence = scaled_features[-window_size:]
last_sequence = np.expand_dims(last_sequence, axis=0)
predicted_scaled = model.predict(last_sequence)
predicted_values = scaler.inverse_transform(predicted_scaled)[0]

# Calculate the mean of the last window_size days for each symptom
historical_mean = data[symptoms].iloc[-window_size:].mean()

# Calculate the difference from the historical trend (mean) and the predicted value
difference = predicted_values - historical_mean

# Alert condition: If the predicted value exceeds the historical mean by 35% for any symptom
alert_triggered = any(
    difference[i] > historical_mean[i] * 1.35
    for i in range(len(symptoms))
)

# Update the alert condition message based on the prediction
if alert_triggered:
    st.error("🚨 A disease outbreak may occur!")
else:
    st.success("👍 No major outbreak trend detected.")

# Graph-only visualization
st.write("### Predicted Symptom Trends for Next Day")

for i, symptom in enumerate(symptoms):
    fig, ax = plt.subplots()
    ax.plot(data['day'], data[symptom], label='Actual', marker='o')
    ax.plot(
        [data['day'].iloc[-1] + 1],
        [predicted_values[i]],
        marker='X',
        markersize=10,
        color='red',
        label='Predicted'
    )
    ax.set_title(f"{symptom.replace('_', ' ').title()}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cases")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
  
st.caption("\U0001F4A1 Made by Siddhanth,Anish,Diagnta,Adrita & Monalisa")
