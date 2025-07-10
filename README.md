This repository contains the official source codes for the paper: "A Novel Adaptive Concept Drift Detection Approach for Evolving Network Traffic Patterns".

Our work introduces an adaptive system that leverages a Variational Autoencoder (VAE) and a dual-detector architecture (Page-Hinkley and ADWIN) to effectively identify and adapt to both sudden and gradual concept drifts in network traffic data.

Key Features

-   Dual-Detector Architecture: Integrates Page-Hinkley for sudden drifts and ADWIN for gradual drifts.
-   Unsupervised Drift Signal: Uses the reconstruction error from a VAE trained on benign traffic as the primary drift indicator.
-   Adaptive Retraining: Triggers model retraining only when significant, persistent drift is detected to maintain performance.
-   Dynamic Window Sizing: Adjusts the data processing window size based on the type of drift detected to balance stability and responsiveness.
-   Reproducible Experiments: Includes scripts to inject controlled drifts and reproduce the results presented in the paper.
