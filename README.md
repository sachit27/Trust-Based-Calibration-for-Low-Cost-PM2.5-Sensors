# Trust-Based Calibration for Low-Cost PM2.5 Sensors

This repository contains the code, data, and resources supporting the paper:

**Dynamic Calibration of Low-Cost PM2.5 Sensors Using Trust-Based Consensus Mechanisms**  
_Sachit Mahajan, Dirk Helbing — ETH Zurich_

## Overview

Low-cost PM2.5 sensors offer the promise of high-resolution air quality monitoring, but suffer from reliability issues due to environmental factors, drift, and manufacturing inconsistencies.

We present a **trust-based adaptive calibration framework** that:

- Corrects systematic sensor errors via offset–scale transformations.
- Dynamically computes a **trust score** per sensor based on four criteria: accuracy, stability, responsiveness, and consensus alignment.
- Adjusts model complexity and calibration strength depending on each sensor’s trust score.
- Utilizes **wavelet-based feature extraction** and **trust-weighted consensus** to achieve scalable, accurate sensor correction.

Our method outperforms traditional techniques (e.g., linear regression, SVR, gradient boosting) in both simulated and real-world deployments, achieving up to **68% MAE reduction** for low-quality sensors and **35–38%** for already reliable ones.

## 🔬 Key Features

- **Offset–Scale Correction:** Aligns raw sensor data with reference-grade baselines.
- **Dynamic Trust Computation:** Quantifies sensor reliability using 4 trust indicators.
- **Wavelet Feature Engineering:** Captures multi-scale pollutant trends and anomalies.
- **Adaptive Model Depth:** Lower-trust sensors receive deeper models automatically.
- **Trust-Weighted Consensus:** Learns from the collective wisdom of nearby sensors.

## 📊 Performance

Our framework consistently improves performance across diverse configurations:
- Real-world test set shows **>65% reduction in MAE** for poorly performing sensors.
- Outperforms Random Forest, SVR, and Gradient Boosting in nearly all conditions.
- Robust to drift, scaling errors, and localized pollution events.

## File Overview
- main.py: Entry point for running calibration on real sensor data.
- src/calibrator.py: Main HybridAdaptiveSensorCalibration class implementing trust-based correction.
- src/evaluation.py: Functions for computing evaluation metrics (MAE, RMSE, correlation, etc.).
- src/simulation.py: Code for generating synthetic datasets and running Monte Carlo simulations.
- combined_data.csv: Example dataset with raw sensor and reference values. You can replace it with your own.

