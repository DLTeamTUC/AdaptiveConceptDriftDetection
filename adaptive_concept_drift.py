# Adaptive_concept_drift.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from river.drift import ADWIN, PageHinkley

# --- CONFIGURATION ---
# These parameters are based on the experimental setup described in the paper.
# --------------------------------------------------------------------------
# File paths
DATASET_PATH = 'Injected_drift_benign_CICIoT2023.csv'
MODEL_PATH = 'vae_model.h5'
THRESHOLD_PATH = 'T1_threshold.npy'

CONFIG = {
    # Page-Hinkley parameters for sudden drift
    "pagehinkley": {
        "threshold": 50,    # Calibrated threshold for significant deviation
        "min_instances": 80, # Min samples before PH is active
    },
    # ADWIN parameters for gradual drift
    "adwin": {
        "delta": 0.001,     # Sensitivity parameter
    },
    # Adaptive window (batch) size parameters
    "window": {
        "min_size": 18000,
        "max_size": 22000,
        "initial_size": 20000,
    },
    # Model retraining parameters
    "training": {
        "epochs": 3,
        "batch_size": 64,
        "validation_split": 0.2,
        "smoothing_factor": 0.1, # For updating the T1 threshold
    },
    # Drift response logic
    "drift": {
        "gradual_retrain_threshold": 4,  # Retrain if 4 of the last 7 windows show gradual drift
        "suppression_windows": 1,        # Skip drift checks for 1 window after retraining
    },
    # General settings
    "logging": {
        "verbose": True  # Set to False to reduce console output
    }
}

# --- GLOBAL STATE TRACKERS ---
# These globals track state across processing windows.
gradual_drift_history = []  # Stores window numbers with detected gradual drift
sudden_drift_events = []    # Stores (window_num, start_idx, end_idx) for sudden drifts
gradual_drift_events = []   # Stores (window_num, start_idx, end_idx) for gradual drifts

# --- HELPER FUNCTIONS ---

def calculate_reconstruction_error(model, data):
    """Computes the mean squared error for each sample."""
    reconstructed = model.predict(data, verbose=0)
    return np.mean(np.square(data - reconstructed), axis=1)

def update_threshold(base_threshold, new_errors):
    """Updates the T1 threshold using exponential smoothing."""
    smoothing_factor = CONFIG["training"]["smoothing_factor"]
    new_quantile_val = np.quantile(new_errors, 0.99)
    return (1 - smoothing_factor) * base_threshold + smoothing_factor * new_quantile_val

def adjust_window_size(is_sudden, is_gradual, current_size):
    """Adjusts the window size based on the detected drift type."""
    min_size = CONFIG["window"]["min_size"]
    max_size = CONFIG["window"]["max_size"]
    
    if is_sudden:
        # For sudden drift, shrink window for faster reaction to subsequent changes.
        new_size = max(min_size, int(current_size * 0.8))
    elif is_gradual:
        # For gradual drift, increase window to build more statistical confidence
        # and avoid overreacting to minor fluctuations.
        new_size = min(max_size, int(current_size * 1.1))
    else:
        # If no drift, slowly revert to the initial size.
        initial_size = CONFIG["window"]["initial_size"]
        if current_size < initial_size:
            new_size = min(initial_size, int(current_size * 1.05))
        elif current_size > initial_size:
            new_size = max(initial_size, int(current_size * 0.95))
        else:
            new_size = current_size
            
    return new_size

def create_drift_detectors():
    """Initializes and returns new instances of drift detectors."""
    ph = PageHinkley(
        threshold=CONFIG["pagehinkley"]["threshold"],
        min_instances=CONFIG["pagehinkley"]["min_instances"]
    )
    adwin = ADWIN(delta=CONFIG["adwin"]["delta"])
    return ph, adwin

def handle_drift_retraining(model, new_data, reference_threshold):
    """Retrains the VAE model on the new data that triggered the drift."""
    print("--- Retraining VAE model on new data distribution ---")
    
    errors_before = calculate_reconstruction_error(model, new_data)
    
    # Retrain the model
    model.fit(new_data, new_data,
              epochs=CONFIG["training"]["epochs"],
              batch_size=CONFIG["training"]["batch_size"],
              validation_split=CONFIG["training"]["validation_split"],
              callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],
              shuffle=True, verbose=0)
    
    errors_after = calculate_reconstruction_error(model, new_data)
    new_threshold = update_threshold(reference_threshold, errors_after)
    
    if CONFIG["logging"]["verbose"]:
        print(f"Retraining complete. Mean error before: {np.mean(errors_before):.4f}, after: {np.mean(errors_after):.4f}")
        print(f"T1 threshold updated from {reference_threshold:.4f} to {new_threshold:.4f}")
        
    return model, new_threshold, errors_before, errors_after

# --- MAIN EXECUTION ---
def main():
    """Main workflow for adaptive drift detection."""
    tf.random.set_seed(42)
    np.random.seed(42)

    # 1. Load pre-trained model, data, and initial threshold
    print("--- Starting Adaptive Drift Detection Process ---")
    try:
        model = load_model(MODEL_PATH)
        reference_threshold = np.load(THRESHOLD_PATH)
        data_stream = pd.read_csv(DATASET_PATH).iloc[130000:].to_numpy() # Start from the streaming part
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Please run train_vae.py and stats_drift_inject.py first.")
        return
        
    print(f"Loaded VAE model, T1 threshold: {reference_threshold:.4f}, and data stream with {len(data_stream)} samples.")
    
    # 2. Initialize detectors and state variables
    ph, adwin = create_drift_detectors()
    window_size = CONFIG["window"]["initial_size"]
    last_retraining_window = -np.inf
    
    drifted_errors_before_list = []
    drifted_errors_after_list = []

    start_idx, window_num = 0, 0
    start_time = time.perf_counter()

    # 3. Main processing loop over the data stream
    while start_idx < len(data_stream):
        window_num += 1
        end_idx = min(start_idx + window_size, len(data_stream))
        window_data = data_stream[start_idx:end_idx]
        
        if len(window_data) < CONFIG["window"]["min_size"]:
            print("Remaining data is smaller than min window size. Ending process.")
            break

        print(f"\n--- Processing Window {window_num} (Indices {start_idx}-{end_idx-1}) ---")
        
        # Calculate reconstruction error for the current window
        window_errors = calculate_reconstruction_error(model, window_data)
        
        # Skip drift detection if in suppression period
        if window_num - last_retraining_window <= CONFIG["drift"]["suppression_windows"]:
            print(f"In suppression period. Skipping drift detection for this window.")
            start_idx += window_size
            continue

        # 4. Feed errors to detectors
        is_sudden_drift = False
        for err in window_errors:
            ph.update(err)
            if ph.drift_detected:
                is_sudden_drift = True
                break # A single detection flags the whole window
        
        is_gradual_drift = False
        if not is_sudden_drift: # Only check for gradual if no sudden drift was found
            adwin.update(np.mean(window_errors)) # ADWIN works best on the mean of the window error
            if adwin.drift_detected:
                is_gradual_drift = True

        # 5. Determine if retraining is needed based on drift signals
        retraining_triggered = False
        drift_type = "None"
        
        if is_sudden_drift:
            retraining_triggered = True
            drift_type = "Sudden"
            print(f"EVENT: Sudden Drift Detected in Window {window_num}.")
            sudden_drift_events.append((window_num, start_idx, end_idx-1))
        
        elif is_gradual_drift:
            print(f"SIGNAL: Gradual Drift Signal in Window {window_num}.")
            gradual_drift_history.append(window_num)
            gradual_drift_events.append((window_num, start_idx, end_idx-1))
            
            # Check if enough recent windows show gradual drift to warrant retraining
            recent_gradual_windows = [w for w in gradual_drift_history if w > window_num - 7]
            if len(recent_gradual_windows) >= CONFIG["drift"]["gradual_retrain_threshold"]:
                retraining_triggered = True
                drift_type = "Gradual (Confirmed)"
                print(f"EVENT: Gradual Drift Confirmed. Triggering retraining.")
                gradual_drift_history = [] # Reset after trigger
        
        # 6. Perform adaptation if triggered
        if retraining_triggered:
            last_retraining_window = window_num
            model, reference_threshold, errs_before, errs_after = handle_drift_retraining(
                model, window_data, reference_threshold
            )
            drifted_errors_before_list.append(errs_before)
            drifted_errors_after_list.append(errs_after)
            
            # Reset detectors to adapt to the new baseline
            ph, adwin = create_drift_detectors()
        
        # 7. Adjust window size for the next iteration
        window_size = adjust_window_size(is_sudden_drift, retraining_triggered and drift_type.startswith("Gradual"), window_size)
        print(f"Next window size set to: {window_size}")
        
        # Move to the next window
        start_idx = end_idx
        
    end_time = time.perf_counter()
    print(f"\n--- Process Finished in {end_time - start_time:.2f} seconds ---")

    # 8. Final Summary
    print("\n--- Final Summary ---")
    print(f"Total windows processed: {window_num}")
    print(f"Detected Sudden Drift Events: {len(sudden_drift_events)}")
    for w, s, e in sudden_drift_events: print(f"  - Window {w} (Indices {s}-{e})")
    print(f"Detected Gradual Drift Signals: {len(gradual_drift_events)}")
    for w, s, e in gradual_drift_events: print(f"  - Window {w} (Indices {s}-{e})")
    
    # 9. Plot results
    if drifted_errors_before_list:
        plt.figure(figsize=(15, 7))
        all_errors_before = np.concatenate(drifted_errors_before_list)
        all_errors_after = np.concatenate(drifted_errors_after_list)
        
        plt.plot(all_errors_before, label='Before Retraining', alpha=0.7)
        plt.plot(all_errors_after, label='After Retraining', alpha=0.9)
        plt.xlabel('Concatenated Sample Index from Drifted Windows')
        plt.ylabel('Reconstruction Error')
        plt.title('Reconstruction Errors Before and After Adaptation')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

if __name__ == "__main__":
    main()