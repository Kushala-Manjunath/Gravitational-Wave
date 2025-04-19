# Detecting Gravitational Waves Using Convolutional Neural Networks

The project leverages Convolutional Neural Networks (CNNs) to detect gravitational waves—faint signals resulting from cosmic events like black hole or neutron star mergers—within noisy data from LIGO detectors. It combines real observational data from the Gravitational Wave Open Science Center (GWOSC) with synthetic waveforms for training and validation.

## Project Overview
The goal is to accurately classify gravitational wave events (BBH, BNS, NSBH) using deep learning. Approach is to transform raw LIGO strain data into time-frequency spectrograms using Q-transform, and train a CNN to classify the signal type. Real-time classification capability with minimal computational overhead compared to traditional matched filtering.

## Dataset

•	Real events from LIGO's O1 & O2 runs (e.g., GW150914, GW170817).

•	150 synthetic waveforms (BBH, BNS, NSBH) generated using PyCBC and injected with LIGO-like noise.

•	Final training set expanded via data augmentation (480 spectrograms).

## Methodology

•	Preprocessing: High-pass filtering, whitening, Q-transform, normalization, resizing to 128×128.

•	Model: CNN with 3 convolutional blocks + dropout + softmax classifier.

•	Training: 80:20 train-validation split, early stopping, class balancing, and dynamic learning rate.

## Results

•	Validation Accuracy: 100% on synthetic data.

•	Real Event Prediction: Detected GW170817 (BNS) with ~90% confidence.

•	Visualizations: Spectrograms, training curves, confusion matrices, and prediction probabilities.

## Tech Stack
Python, TensorFlow, PyCBC, GWpy, Google Colab

## Key Contributions
•	A modular deep learning pipeline for gravitational wave classification.

•	High performance on both synthetic and real data.

•	Potential integration into real-time gravitational wave detection systems.

