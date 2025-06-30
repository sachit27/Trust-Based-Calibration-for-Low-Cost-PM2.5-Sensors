
# main.py
import pandas as pd
from src.calibrator import HybridAdaptiveSensorCalibration
from src.evaluation import evaluate_all_metrics_on_test

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('combined_data.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y %H:%M')
    data = data.sort_values('Date').reset_index(drop=True)
    sensors = ['S1', 'S2', 'S3', 'S4']
    calibrator = HybridAdaptiveSensorCalibration(
        window_size=24,
        error_scale=2.0,
        min_trust_score=0.2,
        sensor_positions={},
        use_spatial=False
    )
    metrics_df = evaluate_all_metrics_on_test(data, sensors, calibrator)
    print(metrics_df.to_string(index=False))
