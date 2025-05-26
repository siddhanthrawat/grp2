import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Title and subtitle
st.title("🧠 Pandemica - Disease Outbreak Predictor")
st.subheader("AI-driven early warning system for infectious disease surveillance")

# Sidebar controls
st.sidebar.header("Simulate or Upload Data")
data_mode = st.sidebar.radio("Choose Input Mode:", ("Simulate Synthetic Data", "Upload CSV"))

# Define symptom columns
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
        'malnutrition_cases': np.random.poisson(lam=4, size=30),
        'conjunctivitis_cases': np.random.poisson(lam=3, size=30),
        'jaundice_cases': np.random.poisson(lam=1, size=30),
    })
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if not all(symptom in data.columns for symptom in symptoms):
            st.error(f"CSV file must contain columns: {', '.join(symptoms)}")
            st.stop()
    else:
        st.warning("Please upload a CSV file with all symptom columns.")
        st.stop()

# Preview data
st.write("### Clinical Data Preview:")
st.dataframe(data.head())

# Window size for moving average
window_size = 5

# Calculate historical mean and last values
historical_mean = data[symptoms].iloc[-window_size:].mean()
latest_values = data[symptoms].iloc[-1]
predicted_values = latest_values * 1.1  # Just for visualization
difference = latest_values - historical_mean

# Alert logic (35% increase in any symptom)
alert_triggered = any(
    difference[i] > historical_mean[i] * 0.35
    for i in range(len(symptoms))
)

# Display alert or status
if alert_triggered:
    st.error("🚨 A disease outbreak may occur!")
else:
    st.success("👍 No major outbreak trend detected.")

# Graph for each symptom
st.write("### Symptom Trends and Next Day Estimates")
for i, symptom in enumerate(symptoms):
    fig, ax = plt.subplots()
    ax.plot(data['day'], data[symptom], label='Actual', marker='o')
    ax.plot([data['day'].iloc[-1] + 1], [predicted_values[i]], label='Next Day Estimate',
            marker='X', markersize=10, color='red')
    ax.set_xlabel("Day")
    ax.set_ylabel(symptom.replace("_", " ").capitalize())
    ax.legend()
    ax.grid(True)
    st.pyplot(fig
