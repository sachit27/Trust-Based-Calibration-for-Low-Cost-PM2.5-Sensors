
"""
calibrator.py

Contains the core class HybridAdaptiveSensorCalibration implementing the trust-based
dynamic calibration framework described in the paper.
"""

import numpy as np
import pandas as pd
import pywt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

class HybridAdaptiveSensorCalibration:
    def __init__(self, window_size=24, error_scale=2.0, min_trust_score=0.2,
                 sensor_positions=None, D=5.0, use_spatial=True):
        self.window_size = window_size
        self.error_scale = error_scale
        self.min_trust_score = min_trust_score
        self.sensor_positions = sensor_positions if sensor_positions else {}
        self.D = D
        self.use_spatial = use_spatial
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.local_models = {}
        self.trust_scores = {}
        self.offsets_ = {}
        self.scales_ = {}

    def fit_offset_scale(self, data, sensors, ref='Ref'):
        for sensor in sensors:
            valid_mask = ~data[[sensor, ref]].isna().any(axis=1)
            x = data.loc[valid_mask, sensor].values
            y = data.loc[valid_mask, ref].values
            if len(x) < 2:
                self.offsets_[sensor] = 0.0
                self.scales_[sensor] = 1.0
                continue
            slope, intercept, _, _, _ = stats.linregress(x, y)
            b_i = slope if not np.isclose(slope, 0) else 1e-6
            a_i = -intercept / b_i
            self.offsets_[sensor] = a_i
            self.scales_[sensor] = b_i

    def apply_offset_scale(self, data, sensors):
        df = data.copy()
        for sensor in sensors:
            a_i = self.offsets_.get(sensor, 0.0)
            b_i = self.scales_.get(sensor, 1.0)
            df[sensor] = b_i * (df[sensor] - a_i)
        return df

    def compute_trust_score(self, data, sensor, ref='Ref'):
        errors = np.abs(data[sensor] - data[ref])
        accuracy_score = np.mean(np.exp(-errors / self.error_scale))
        rolling_std = data[sensor].rolling(window=self.window_size, min_periods=1).std()
        denom = data[sensor].mean() if data[sensor].mean() != 0 else 1e-6
        stability_score = np.mean(np.exp(-rolling_std / denom))
        try:
            responsiveness_score = abs(stats.pearsonr(data[sensor].diff().fillna(0),
                                                      data[ref].diff().fillna(0))[0])
        except:
            responsiveness_score = 0.0
        other_cols = [c for c in data.columns if c not in [sensor, 'Date']]
        consensus_series = data[other_cols].mean(axis=1)
        try:
            corr_val = stats.pearsonr(data[sensor], consensus_series)[0]
        except:
            corr_val = 0.0
        consensus_score = np.exp(-abs(corr_val - 1))
        trust_raw = (accuracy_score + stability_score +
                     responsiveness_score + consensus_score) / 4.0
        return max(trust_raw, self.min_trust_score)

    def _extract_wavelet_features(self, series, wavelet='db1', level=3):
        coeffs = pywt.wavedec(series, wavelet, level=level)
        feats = {}
        for i, cff in enumerate(coeffs):
            feats[f'wave_mean_{i}'] = np.mean(np.abs(cff))
            feats[f'wave_std_{i}']  = np.std(cff)
            feats[f'wave_max_{i}']  = np.max(np.abs(cff))
        return feats

    def _build_spatial_consensus(self, data, sensor, sensors):
        if not self.use_spatial or sensor not in self.sensor_positions:
            return data[[s for s in sensors if s != sensor]].mean(axis=1)
        (xi, yi) = self.sensor_positions[sensor]
        w_sum = 0.0
        weighted_val = pd.Series(0.0, index=data.index)
        for s_other in sensors:
            if s_other == sensor:
                continue
            if s_other not in self.sensor_positions:
                w = 1.0
            else:
                (xj, yj) = self.sensor_positions[s_other]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                w = np.exp(-dist/self.D)
            weighted_val += w * data[s_other]
            w_sum += w
        if w_sum == 0:
            return data[[s for s in sensors if s != sensor]].mean(axis=1)
        return weighted_val / w_sum

    def build_advanced_features(self, data, sensor, sensors, trust_scores=None, ref='Ref'):
        feats = self._extract_wavelet_features(data[sensor].fillna(method='ffill'))
        feats['roll_mean'] = data[sensor].rolling(self.window_size, min_periods=1).mean()
        feats['roll_std'] = data[sensor].rolling(self.window_size, min_periods=1).std()
        feats['spatial_consensus'] = self._build_spatial_consensus(data, sensor, sensors)
        feats['sensor_val'] = data[sensor]
        feats['hour'] = pd.to_datetime(data['Date']).dt.hour.fillna(0)
        feat_df = pd.DataFrame(feats, index=data.index)
        for other in sensors:
            if other != sensor:
                corrs = data[sensor].rolling(self.window_size).corr(data[other]).fillna(0)
                feat_df[f'corr_{other}'] = corrs
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
        return feat_df

    def build_adaptive_model(self, trust_score):
        return GradientBoostingRegressor(
            n_estimators=int(100 + (1 - trust_score)*100),
            learning_rate=max(0.01, trust_score*0.1),
            max_depth=int(2 + (1 - trust_score)*2),
            subsample=0.8 if trust_score > 0.5 else 0.7,
            min_samples_leaf=int(10 * trust_score),
            random_state=42
        )

    def fit_transform(self, data, sensors, ref='Ref'):
        df = data.copy()
        self.fit_offset_scale(df, sensors, ref)
        df_corrected = self.apply_offset_scale(df, sensors)
        calibrated = {}
        for s in sensors:
            trust = self.compute_trust_score(df_corrected, s, ref)
            self.trust_scores[s] = trust
            feats_df = self.build_advanced_features(df_corrected, s, sensors, self.trust_scores, ref)
            arr_imp = self.imputer.fit_transform(feats_df)
            arr_scl = self.scaler.fit_transform(arr_imp)
            model = self.build_adaptive_model(trust)
            model.fit(arr_scl, df_corrected[ref])
            preds = model.predict(arr_scl)
            final_vals = min(0.8, trust)*df_corrected[s] + (1 - min(0.8, trust))*preds
            calibrated[s] = final_vals
        return pd.DataFrame(calibrated, index=df_corrected.index), self.trust_scores
