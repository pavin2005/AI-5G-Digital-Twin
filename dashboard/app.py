import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# --- Page Config ---
st.set_page_config(page_title="5G Digital Twin", layout="wide")
st.title("📡 AI-Powered 5G Digital Twin")
st.markdown("Monitor network conditions and simulate 'What-If' scenarios using Machine Learning.")

# --- Load Data & Train Model ---
@st.cache_data
def load_and_train():
    df = pd.read_csv("data/sample_data.csv")
    # Features and Target
    X = df[['ue_count', 'traffic_mbps', 'resource_utilization']]
    y = df['latency_ms']
    
    # Train
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return df, model

df, model = load_and_train()

# --- Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Current Network State (Live KPIs)")
    st.dataframe(df.tail(5)) # Show latest 5 records
    
    # Simple alert logic
    avg_latency = df['latency_ms'].mean()
    if avg_latency > 30:
        st.warning(f"⚠ Warning: Average network latency is high ({avg_latency:.1f} ms)")
    else:
        st.success(f"✅ Network operating normally ({avg_latency:.1f} ms)")

with col2:
    st.subheader("🧪 What-If Simulation Engine")
    st.markdown("Adjust the parameters below to predict network latency.")
    
    # Sliders for user input
    sim_ue = st.slider("Number of Users (UEs)", min_value=10, max_value=1500, value=500, step=10)
    sim_traffic = st.slider("Network Traffic (Mbps)", min_value=10, max_value=1000, value=200, step=10)
    sim_resources = st.slider("Resource Utilization (%)", min_value=10, max_value=100, value=60, step=1)
    
    # Prediction
    scenario_data = pd.DataFrame({
        'ue_count': [sim_ue],
        'traffic_mbps': [sim_traffic],
        'resource_utilization': [sim_resources]
    })
    
    predicted_lat = model.predict(scenario_data)[0]
    
    st.markdown("### 🤖 AI Prediction")
    if predicted_lat > 40:
        st.error(f"**Predicted Latency: {predicted_lat:.2f} ms (Severe Congestion)**")
    elif predicted_lat > 25:
        st.warning(f"**Predicted Latency: {predicted_lat:.2f} ms (Moderate Load)**")
    else:
        st.success(f"**Predicted Latency: {predicted_lat:.2f} ms (Optimal QoS)**")