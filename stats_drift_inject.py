# stats_drift_inject.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, ks_2samp
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
# This script is configured to reproduce EXPERIMENT 4 from the paper.
# To reproduce other experiments, modify the `SUDDEN_DRIFT_INDICES` and `GRADUAL_DRIFT_INTERVALS`.
# --------------------------------------------------------------------------
# File paths
INPUT_DATASET = 'benign_CICIoT2023.csv'
OUTPUT_DATASET = 'Injected_drift_benign_CICIoT2023.csv'

# Drift Injection Parameters
DRIFT_WINDOW_SIZE = 20000          # The size of each data batch/window
TOP_FEATURES_TO_INJECT = 10        # Number of features to inject drift into
SUDDEN_DRIFT_STD_MULTIPLIER = 5    # How strongly to inject sudden drift (scaled by feature std dev)
GRADUAL_DRIFT_INCREMENT = 0.05     # How much to increment drift in each segment of a gradual drift window
SUDDEN_DRIFT_AFFECTED_FRACTION = 0.8 # Percentage of flows within a segment to alter for sudden drift

# Drift Injection locations for EXPERIMENT 4
SUDDEN_DRIFT_INDICES = [3, 38]
GRADUAL_DRIFT_INTERVALS = [(10, 25)] # List of (start_index, end_index) tuples

# Analysis Parameters
DETECTION_WINDOW_SIZE = 2000       # Window size for pre-analysis of drift sensitivity

# --- DRIFT ANALYSIS FUNCTIONS ---

def compute_kl_divergence(p, q):
    """Compute KL divergence, ensuring non-zero probabilities."""
    p = np.asarray(p, dtype=np.float64) + 1e-10
    q = np.asarray(q, dtype=np.float64) + 1e-10
    return entropy(p, q)

def compute_ks_test(reference_window, current_window):
    """Compute the Kolmogorov-Smirnov (KS) statistic."""
    ks_stat, _ = ks_2samp(reference_window, current_window)
    return ks_stat

def analyze_feature_drift(dataset, feature_names, window_size):
    """Slide a window to compute drift metrics (KL, KS) for each feature."""
    drift_results = []
    for feature in feature_names:
        reference_window = dataset[feature][:window_size].values
        for start in range(window_size, len(dataset), window_size):
            current_window = dataset[feature][start:start + window_size].values
            if len(current_window) < window_size:
                break
            
            kl_div = compute_kl_divergence(reference_window, current_window)
            ks_stat = compute_ks_test(reference_window, current_window)
            drift_results.append({
                "feature": feature, "window_start": start,
                "kl_divergence": kl_div, "ks_stat": ks_stat
            })
            reference_window = current_window
    return pd.DataFrame(drift_results)

# --- DRIFT INJECTION FUNCTIONS ---

def inject_drift(df, drift_summary):
    """Injects both sudden and gradual drifts into the provided dataframe."""
    print("--- Starting Drift Injection ---")
    
    # 1. Identify most drift-sensitive features from the analysis
    if not drift_summary.empty:
        important_features = drift_summary.index[:TOP_FEATURES_TO_INJECT].tolist()
    else: # Fallback if analysis is empty
        important_features = df.columns[:TOP_FEATURES_TO_INJECT].tolist()
    print(f"Top {len(important_features)} features selected for drift injection: {important_features}")
    
    # 2. Define historical and streaming data splits
    initial_training_size = 130000
    historical_data = df.iloc[:initial_training_size].copy()
    data_stream = df.iloc[initial_training_size:].copy()
    
    # Compute baseline statistics from the stable historical part
    baseline_stats = {
        feature: (historical_data[feature].mean(), historical_data[feature].std())
        for feature in important_features
    }
    
    # 3. Split the data stream into non-overlapping windows for injection
    windows = [
        data_stream.iloc[i:i + DRIFT_WINDOW_SIZE].copy()
        for i in range(0, len(data_stream), DRIFT_WINDOW_SIZE)
        if len(data_stream.iloc[i:i + DRIFT_WINDOW_SIZE]) == DRIFT_WINDOW_SIZE
    ]
    print(f"Data stream split into {len(windows)} windows of size {DRIFT_WINDOW_SIZE}.")

    # 4. Inject Sudden Drift
    print(f"Injecting sudden drift into windows: {SUDDEN_DRIFT_INDICES}")
    for idx in SUDDEN_DRIFT_INDICES:
        if idx < len(windows):
            window = windows[idx]
            seg_start = len(window) // 2 - 2500
            seg_end = seg_start + 5000
            indices_segment = window.iloc[seg_start:seg_end].index
            
            # Select a random block within the segment to modify
            block_size = int(len(indices_segment) * SUDDEN_DRIFT_AFFECTED_FRACTION)
            start_idx_block = np.random.randint(0, len(indices_segment) - block_size + 1)
            selected_indices = indices_segment[start_idx_block : start_idx_block + block_size]
            
            for feature in important_features:
                mean_val, std_val = baseline_stats.get(feature, (1, 0))
                # The drift factor is proportional to the feature's relative volatility
                drift_factor = 1 + SUDDEN_DRIFT_STD_MULTIPLIER * (std_val / (mean_val + 1e-9))
                window.loc[selected_indices, feature] *= drift_factor
            windows[idx] = window # Update the list of windows
        else:
            print(f"Warning: Sudden drift window index {idx} is out of range.")
            
    # 5. Inject Gradual Drift
    print(f"Injecting gradual drift in intervals: {GRADUAL_DRIFT_INTERVALS}")
    for (start_interval, end_interval) in GRADUAL_DRIFT_INTERVALS:
        for idx in range(start_interval, end_interval + 1):
            if idx < len(windows):
                window = windows[idx]
                n_segments = 5
                seg_length = len(window) // n_segments
                for seg_num in range(n_segments):
                    # Drift factor increases with each segment
                    drift_factor = 1 + GRADUAL_DRIFT_INCREMENT * (seg_num + 1)
                    seg_start = seg_num * seg_length
                    seg_end = (seg_num + 1) * seg_length
                    window.iloc[seg_start:seg_end] = window.iloc[seg_start:seg_end] * drift_factor
                windows[idx] = window
            else:
                print(f"Warning: Gradual drift window index {idx} is out of range.")

    # 6. Recombine and save the final dataset
    final_data = pd.concat([historical_data] + windows, ignore_index=True)
    print("Final dataset with injected drift prepared.")
    final_data.to_csv(OUTPUT_DATASET, index=False)
    print(f"Drift-injected dataset saved to '{OUTPUT_DATASET}'.")
    return final_data

# --- VISUALIZATION FUNCTIONS ---
def plot_drift_summary(drift_summary):
    # (Visualization functions remain as they are, no changes needed)
    pass 

# --- MAIN EXECUTION ---
def main():
    """Main workflow to load, analyze, inject drift, and save data."""
    print("--- Starting Drift Injection Process ---")
    
    # 1. Load and clean the dataset
    print(f"Loading and cleaning data from {INPUT_DATASET}...")
    try:
        data = pd.read_csv(INPUT_DATASET)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{INPUT_DATASET}'. Please place it in the directory.")
        return

    data.columns = data.columns.str.strip()
    
    # Use a set for efficient removal of duplicate column names
    columns_to_remove = set([
        'id', 'Flow ID', 'Src IP', 'Src Port', 'ARP', 'Weight', 'ack_count', 
        'fin_flag_number', 'Duration', 'rst_flag_number', 'syn_flag_number', 
        'syn_count', 'Protocol Type', 'fin_count', 'Min','Srate', 'Rate',
        'LLC','IPv', 'DNS', 'ICMP', 'label'
    ])
    
    data.drop(columns=list(columns_to_remove), errors='ignore', inplace=True)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    print(f"Data loaded and cleaned. Shape: {data.shape}")
    
    # 2. Scale data using MinMaxScaler
    print("Scaling data...")
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    
    # 3. Perform initial drift analysis to identify sensitive features
    print("Running initial drift analysis...")
    _, drift_summary = analyze_feature_drift(data, data.columns, window_size=DETECTION_WINDOW_SIZE)
    
    # 4. Inject controlled drifts into the dataset
    inject_drift(data, drift_summary)
    
    print("--- Process Completed Successfully ---")

if __name__ == '__main__':
    main()