# src/evaluation.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

def compute_full_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    corr = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else np.nan
    nrmse = rmse / np.mean(y_true) * 100
    norm_bias = bias / np.mean(y_true) * 100
    return [mae, rmse, corr, bias, nrmse, norm_bias]

def evaluate_all_metrics_on_test(data, sensors, calibrator, ref='Ref', train_ratio=0.7):
    df = data.copy().sort_values('Date').reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    train_calibrated, _ = calibrator.fit_transform(train_df, sensors, ref)
    test_calibrated, _ = calibrator.fit_transform(test_df, sensors, ref)
    all_results = []
    imp = SimpleImputer(strategy='mean')
    for s in sensors:
        y_test = test_df[ref].values
        X_test = pd.DataFrame({'val': test_df[s], 'hour': pd.to_datetime(test_df['Date']).dt.hour})
        X_imp = imp.fit_transform(X_test)
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
        for name, model in models.items():
            X_train = pd.DataFrame({'val': train_df[s], 'hour': pd.to_datetime(train_df['Date']).dt.hour})
            y_train = train_df[ref].values
            X_train_imp = imp.fit_transform(X_train)
            model.fit(X_train_imp, y_train)
            preds = model.predict(X_imp)
            row = compute_full_metrics(y_test, preds)
            all_results.append([s, name] + row)
        hybrid_preds = test_calibrated[s].values
        row = compute_full_metrics(y_test, hybrid_preds)
        all_results.append([s, 'Hybrid'] + row)
    return pd.DataFrame(all_results, columns=[
        'Sensor', 'Method', 'MAE (µg/m³)', 'RMSE (µg/m³)',
        'Correlation', 'Bias (µg/m³)', 'NRMSE (%)', 'Norm. Bias (%)'
    ])
