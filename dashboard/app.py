import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest

# --- Page Config ---
st.set_page_config(page_title="5G Digital Twin", layout="wide")
st.title("📡 AI-Powered 5G Digital Twin")
st.markdown("Monitor network conditions, simulate scenarios, and detect hidden anomalies.")

# --- Load Data & Train Models ---
@st.cache_data
def load_and_train():
    df = pd.read_csv("data/sample_data.csv")
    
    # Features for models
    X = df[['ue_count', 'traffic_mbps', 'resource_utilization']]
    y_latency = df['latency_ms']
    
    # 1. Train Predictive Model (Random Forest)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y_latency)
    
    # 2. Train Anomaly Model (Isolation Forest)
    iso_model = IsolationForest(contamination=0.1, random_state=42)
    iso_model.fit(X)
    
    return df, rf_model, iso_model

df, rf_model, iso_model = load_and_train()

# --- Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Current Network State (Live KPIs)")
    st.dataframe(df.tail(6)) 
    
    avg_latency = df['latency_ms'].mean()
    if avg_latency > 30:
        st.warning(f"⚠ Warning: Average network latency is high ({avg_latency:.1f} ms)")
    else:
        st.success(f"✅ Network operating normally ({avg_latency:.1f} ms)")

with col2:
    st.subheader("🧪 What-If Simulation Engine")
    st.markdown("Adjust the parameters below to predict latency and detect anomalies.")
    
    sim_ue = st.slider("Number of Users (UEs)", min_value=10, max_value=1500, value=500, step=10)
    sim_traffic = st.slider("Network Traffic (Mbps)", min_value=10, max_value=1000, value=200, step=10)
    sim_resources = st.slider("Resource Utilization (%)", min_value=10, max_value=100, value=60, step=1)
    
    scenario_data = pd.DataFrame({
        'ue_count': [sim_ue],
        'traffic_mbps': [sim_traffic],
        'resource_utilization': [sim_resources]
    })
    
    # Get Predictions
    predicted_lat = rf_model.predict(scenario_data)[0]
    anomaly_score = iso_model.predict(scenario_data)[0] # Returns -1 for anomaly, 1 for normal
    
    st.markdown("### 🤖 AI Prediction Results")
    
    # Latency Output
    if predicted_lat > 40:
        st.error(f"**Predicted Latency: {predicted_lat:.2f} ms (Severe Congestion)**")
    elif predicted_lat > 25:
        st.warning(f"**Predicted Latency: {predicted_lat:.2f} ms (Moderate Load)**")
    else:
        st.success(f"**Predicted Latency: {predicted_lat:.2f} ms (Optimal QoS)**")
        
    # Anomaly Output
    if anomaly_score == -1:
        st.error("🚨 **ANOMALY DETECTED**: This configuration represents highly abnormal network behavior.")
    else:
        st.success("✅ **Status Normal**: Network configuration is within expected operational parameters.")