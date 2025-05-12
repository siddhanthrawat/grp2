import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("\U0001F9A0 SwasthyaNet AI - Disease Outbreak Predictor")
st.subheader("AI-driven early warning system for infectious disease surveillance")

st.sidebar.header("Simulate or Upload Data")
data_mode = st.sidebar.radio("Choose Input Mode:", ("Simulate Synthetic Data", "Upload CSV"))

symptoms = ['fever_cases', 'rash_cases', 'platelet_alerts', 'malnutrition_cases', 'conjunctivitis_cases', 'jaundice_cases']

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
        st.warning("Please upload a CSV file with required columns.")
        st.stop()

st.write("### Clinical Data Preview:")
st.dataframe(data.head())

# Normalize features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[symptoms])

# Create sequences for LSTM (still predicting fever_cases)
X, y = [], []
window_size = 5
for i in range(len(scaled_features) - window_size):
    X.append(scaled_features[i:i+window_size])
    y.append(scaled_features[i+window_size][0])  # Predict fever_cases
X, y = np.array(X), np.array(y)

# LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

# Prediction
last_sequence = scaled_features[-window_size:]
last_sequence = np.expand_dims(last_sequence, axis=0)
predicted_scaled = model.predict(last_sequence)
predicted_fever = scaler.inverse_transform(
    np.concatenate([predicted_scaled, np.zeros((1, len(symptoms)-1))], axis=1)
)[0][0]

st.success(f"Predicted Fever Cases for Day {int(data['day'].max()) + 1}: {predicted_fever:.2f}")

# Outbreak alert
if predicted_fever > data['fever_cases'].iloc[-1] * 1.2:
    st.error("\U0001F6A8 ALERT: Potential Outbreak Predicted!")
else:
    st.info("\U0001F44D No major outbreak trend detected.")

# Visualization
st.write("### Fever Cases vs Predicted")
fig, ax = plt.subplots()
ax.plot(data['day'], data['fever_cases'], label='Actual Fever Cases', marker='o')
ax.plot([data['day'].iloc[-1] + 1], [predicted_fever],
        label='Predicted Next Day', marker='X', markersize=10, color='red')
ax.set_xlabel("Day")
ax.set_ylabel("Fever Cases")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Additional symptom graphs
symptom_labels = {
    'malnutrition_cases': "Malnutrition",
    'conjunctivitis_cases': "Conjunctivitis",
    'rash_cases': "Rashes",
    'jaundice_cases': "Jaundice"
}

st.write("### Symptom Trends")
for symptom_key, symptom_label in symptom_labels.items():
    fig, ax = plt.subplots()
    ax.plot(data['day'], data[symptom_key], marker='o')
    ax.set_title(f"{symptom_label} Over Time")
    ax.set_xlabel("Day")
    ax.set_ylabel(symptom_label)
    ax.grid(True)
    st.pyplot(fig)

st.caption("\U0001F4A1 SwasthyaNet AI - Smart Surveillance for a Healthier Future")
