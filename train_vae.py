# train_vae.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIGURATION ---
CLEAN_DATASET_PATH = 'benign_CICIoT2023.csv'
MODEL_SAVE_PATH = 'vae_model.h5'
THRESHOLD_SAVE_PATH = 'T1_threshold.npy'

# VAE parameters from the paper and your script
INPUT_SHAPE = None # Will be determined from data
INTERMEDIATE_DIM = 512
LATENT_DIM = 30
EPOCHS = 20
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2

# --- VAE MODEL DEFINITION (from your script) ---

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian."""
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_vae(input_shape, latent_dim, intermediate_dim=512):
    """Builds the Variational Autoencoder model using the paper's architecture."""
    # Encoder
    inputs = Input(shape=input_shape)
    x = Dense(intermediate_dim, activation='relu')(inputs)
    x = BatchNormalization()(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x_dec = Dense(intermediate_dim, activation='relu')(latent_inputs)
    x_dec = BatchNormalization()(x_dec)
    outputs = Dense(input_shape[0], activation='sigmoid')(x_dec)
    decoder = Model(latent_inputs, outputs, name='decoder')

    # VAE model
    outputs_vae = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs_vae, name='vae')

    # Define VAE loss
    reconstruction_loss = mse(inputs, outputs_vae) * input_shape[0]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1) * -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    vae.compile(optimizer='adam')
    return vae, encoder, decoder

# --- MAIN SCRIPT ---
def main():
    """Main function to load clean data, train the VAE, and save artifacts."""
    print("--- Starting VAE Training ---")
    
    # 1. Load and preprocess clean benign data
    print(f"Loading clean benign data from {CLEAN_DATASET_PATH}...")
    try:
        # Assuming the clean data has already been scaled and cleaned as in stats_drift_inject.py
        data = pd.read_csv(CLEAN_DATASET_PATH)
        # Drop non-feature columns if they exist
        data.drop(columns=['label'], errors='ignore', inplace=True)
    except FileNotFoundError:
        print(f"Error: Clean dataset file not found at '{CLEAN_DATASET_PATH}'.")
        print("Please ensure the preprocessed, clean benign dataset is available.")
        return
        
    # Use the first 130,000 samples for stable training, as per the paper's methodology
    training_data = data.iloc[:130000].values
    global INPUT_SHAPE
    INPUT_SHAPE = (training_data.shape[1],)
    print(f"Using {len(training_data)} samples for training with input shape {INPUT_SHAPE}.")
    
    # 2. Build and train the VAE
    print("Building VAE model...")
    vae, _, _ = build_vae(INPUT_SHAPE, LATENT_DIM, INTERMEDIATE_DIM)
    vae.summary()
    
    print(f"Training VAE for {EPOCHS} epochs...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    vae.fit(training_data,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=[early_stopping],
            verbose=1)
            
    # 3. Save the trained model
    print(f"Saving trained VAE model to {MODEL_SAVE_PATH}...")
    vae.save(MODEL_SAVE_PATH)
    
    # 4. Calculate and save the T1 threshold
    print("Calculating T1 anomaly threshold...")
    reconstructed = vae.predict(training_data, verbose=0)
    # Using Mean Absolute Error for consistency with some literature, but MSE is also fine.
    reconstruction_errors = np.mean(np.abs(training_data - reconstructed), axis=1)
    
    # T1 is the 99th percentile of reconstruction errors on benign training data
    t1_threshold = np.quantile(reconstruction_errors, 0.99)
    print(f"Calculated T1 threshold (99th percentile): {t1_threshold}")
    
    print(f"Saving threshold to {THRESHOLD_SAVE_PATH}...")
    np.save(THRESHOLD_SAVE_PATH, t1_threshold)
    
    print("--- VAE Training and Setup Complete ---")

if __name__ == '__main__':
    main()