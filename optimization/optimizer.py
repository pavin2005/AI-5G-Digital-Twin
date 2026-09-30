import pandas as pd

def evaluate_mitigation_strategies(rf_model, current_ue, current_traffic, current_resources):
    """
    Tests multiple simulated reconfigurations and selects the one
    that yields the lowest predicted latency.
    """
    strategies = [
        {
            "Strategy": "Baseline (No Action)",
            "ue_count": current_ue,
            "traffic_mbps": current_traffic,
            "resource_utilization": current_resources,
            "Action": "Keep current state"
        },
        {
            "Strategy": "Offload UEs (Handover to Neighbor Cell)",
            "ue_count": max(10, int(current_ue * 0.7)),
            "traffic_mbps": max(10, int(current_traffic * 0.75)),
            "resource_utilization": max(10, int(current_resources * 0.75)),
            "Action": f"Hand over ~{int(current_ue * 0.3)} UEs to adjacent cell"
        },
        {
            "Strategy": "Traffic Throttling / Rate Limiting",
            "ue_count": current_ue,
            "traffic_mbps": max(10, int(current_traffic * 0.6)),
            "resource_utilization": max(10, int(current_resources * 0.7)),
            "Action": "Cap heavy background data streams"
        },
        {
            "Strategy": "Dynamic Bandwidth Reallocation",
            "ue_count": current_ue,
            "traffic_mbps": current_traffic,
            "resource_utilization": max(10, int(current_resources * 0.55)),
            "Action": "Allocate additional PRBs (Physical Resource Blocks)"
        }
    ]

    results = []
    for strat in strategies:
        eval_df = pd.DataFrame([{
            'ue_count': strat['ue_count'],
            'traffic_mbps': strat['traffic_mbps'],
            'resource_utilization': strat['resource_utilization']
        }])
        pred_lat = rf_model.predict(eval_df)[0]
        results.append({
            "Strategy": strat["Strategy"],
            "Predicted Latency (ms)": round(pred_lat, 2),
            "Resource Usage (%)": strat["resource_utilization"],
            "Recommended Action": strat["Action"]
        })

    results_df = pd.DataFrame(results)
    
    # Sort by lowest latency to find the best configuration
    best_option = results_df.sort_values(by="Predicted Latency (ms)").iloc[0]
    return results_df, best_option