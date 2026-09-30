import streamlit as st
import pandas as pd
import sys
import os

# Add root folder to sys.path so we can import from optimization/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimization.optimizer import evaluate_mitigation_strategies

from sklearn.ensemble import RandomForestRegressor, IsolationForest

st.set_page_config(page_title="5G Digital Twin", layout="wide")
st.title("📡 AI-Powered 5G Digital Twin")
st.markdown("Predictive QoS Analysis, Anomaly Detection & Automated What-If Optimization")

@st.cache_data
def load_and_train():
    df = pd.read_csv("data/sample_data.csv")
    X = df[['ue_count', 'traffic_mbps', 'resource_utilization']]
    y = df['latency_ms']
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    iso_model = IsolationForest(contamination=0.1, random_state=42)
    iso_model.fit(X)
    
    return df, rf_model, iso_model

df, rf_model, iso_model = load_and_train()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Current Network State (Live Telemetry)")
    st.dataframe(df.tail(6), use_container_width=True)
    
    avg_latency = df['latency_ms'].mean()
    if avg_latency > 35:
        st.error(f"🚨 Network Alert: Average latency degraded ({avg_latency:.1f} ms)")
    else:
        st.success(f"✅ Network QoS Nominal ({avg_latency:.1f} ms)")

with col2:
    st.subheader("🧪 What-If Simulation Engine")
    sim_ue = st.slider("Simulated UEs (Users)", 10, 1500, 850, step=10)
    sim_traffic = st.slider("Simulated Traffic (Mbps)", 10, 1000, 650, step=10)
    sim_resources = st.slider("Resource Utilization (%)", 10, 100, 92, step=1)
    
    scenario_data = pd.DataFrame([{
        'ue_count': sim_ue,
        'traffic_mbps': sim_traffic,
        'resource_utilization': sim_resources
    }])
    
    predicted_lat = rf_model.predict(scenario_data)[0]
    anomaly_status = iso_model.predict(scenario_data)[0]

    st.markdown("### 🤖 Prediction Summary")
    if predicted_lat > 40:
        st.error(f"**Predicted Latency: {predicted_lat:.2f} ms (Severe Congestion)**")
    elif predicted_lat > 25:
        st.warning(f"**Predicted Latency: {predicted_lat:.2f} ms (Elevated Delay)**")
    else:
        st.success(f"**Predicted Latency: {predicted_lat:.2f} ms (Optimal)**")

    if anomaly_status == -1:
        st.error("🚨 **Anomaly Detected:** Profile deviates from expected operational envelope.")

st.markdown("---")
st.subheader("⚙️ Automated Optimization & Reconfiguration Engine")

if predicted_lat > 30 or anomaly_status == -1:
    st.warning("⚠️ High latency or anomaly detected. Running optimization routines...")
    options_df, best_solution = evaluate_mitigation_strategies(
        rf_model, sim_ue, sim_traffic, sim_resources
    )
    
    st.table(options_df)
    
    st.success(
        f"🎯 **Best Recommended Action:** {best_solution['Strategy']}\n\n"
        f"**Implementation:** {best_solution['Recommended Action']}\n\n"
        f"**Target Latency:** {best_solution['Predicted Latency (ms)']} ms (Resource Usage: {best_solution['Resource Usage (%)']}%)"
    )
else:
    st.info("Performance is optimal. Optimization engine standby.")