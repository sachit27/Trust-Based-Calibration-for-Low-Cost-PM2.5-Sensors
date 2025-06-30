# src/simulation.py
import numpy as np
import pandas as pd
import datetime
from scipy.spatial.distance import cdist

def assign_sensor_positions(num_sensors, layout='grid'):
    positions = {}
    if layout == 'grid':
        grid_size = int(np.ceil(np.sqrt(num_sensors)))
        xs = np.linspace(0, grid_size - 1, grid_size)
        ys = np.linspace(0, grid_size - 1, grid_size)
        xx, yy = np.meshgrid(xs, ys)
        coords = list(zip(xx.flatten(), yy.flatten()))
        for i in range(num_sensors):
            positions[f'S{i+1}'] = coords[i]
    elif layout == 'random':
        coords = np.random.rand(num_sensors, 2) * 10
        for i in range(num_sensors):
            positions[f'S{i+1}'] = tuple(coords[i])
    elif layout == 'cluster':
        center = np.array([5, 5])
        coords = center + np.random.randn(num_sensors, 2)
        for i in range(num_sensors):
            positions[f'S{i+1}'] = tuple(coords[i])
    else:
        raise ValueError("Unsupported layout type. Choose from 'grid', 'random', or 'cluster'.")
    return positions

def generate_synthetic_data_sim(num_sensors=4, num_samples=240, 
                                base_conc=20.0, amplitude=5.0,
                                daily_period=24, temp_range=(10,30),
                                humi_range=(30,80),
                                drift_params={'linear':1.0, 'exp':0.0},
                                noise_params={'sensor':2.0, 'env':1.0},
                                random_seed=42,
                                layout='grid'):
    np.random.seed(random_seed)
    start_time = datetime.datetime(2021, 1, 1)
    dt = pd.date_range(start=start_time, periods=num_samples, freq='1H')
    t = np.arange(num_samples)
    base = base_conc + amplitude * (np.sin(2*np.pi*t/daily_period) + 0.3*np.sin(2*np.pi*t/(daily_period*7)))
    temp = np.linspace(temp_range[0], temp_range[1], num_samples) + np.random.normal(0,2,num_samples)
    humi = np.linspace(humi_range[0], humi_range[1], num_samples) + np.random.normal(0,5,num_samples)
    drift = drift_params['linear'] * t/24 + drift_params['exp'] * np.exp(0.1*t/24)
    ref = base + 0.1*temp - 0.05*humi + drift
    sensor_data = {}
    positions = assign_sensor_positions(num_sensors, layout)
    dist_matrix = cdist(list(positions.values()), list(positions.values()))
    spatial_corr = np.exp(-dist_matrix/5)
    for i in range(num_sensors):
        env_effect = 0.15*temp - 0.07*humi
        offset = np.random.uniform(-3, 3)
        scale = np.random.uniform(0.8, 1.2)
        sensor_drift = np.random.normal(1, 0.1)*drift_params['linear']*t/24
        spatial_noise = spatial_corr[i] @ np.random.normal(0, noise_params['env'], (num_sensors, num_samples))
        raw = scale * (ref + offset + env_effect + sensor_drift) + spatial_noise
        raw += np.random.normal(0, noise_params['sensor'], num_samples) * np.abs(ref)**0.5
        sensor_data[f'S{i+1}'] = raw
    df = pd.DataFrame({'Date': dt, 'Ref': ref, 'Temp': temp, 'Humi': humi, **sensor_data})
    return df, positions
