import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the KPI data
data_path = "data/sample_data.csv"
df = pd.read_csv(data_path)

print("Dataset loaded successfully:")
print(df.head())

# 2. Select Features (inputs) and Target (what we want to predict)
# We want to predict network latency based on load and resource demand
features = ['ue_count', 'traffic_mbps', 'resource_utilization']
target = 'latency_ms'

X = df[features]
y = df[target]

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- Model Performance ---")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# 6. Test a What-If Scenario
# Example: What happens to latency if users jump to 950 with 650 Mbps traffic and 98% resource usage?
scenario = pd.DataFrame({
    'ue_count': [950],
    'traffic_mbps': [650],
    'resource_utilization': [98]
})

predicted_latency = model.predict(scenario)
print("\n--- What-If Scenario Evaluation ---")
print(f"Inputs: 950 UEs, 650 Mbps, 98% Resource Utilization")
print(f"Predicted Latency: {predicted_latency[0]:.2f} ms")