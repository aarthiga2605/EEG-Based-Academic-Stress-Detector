# EEG-Based-Academic-Stress-Detector
Programming Language: MATLAB

This mini project uses EEG signals from the SAM40 dataset to detect academic stress by extracting EEG-based biomarkers and classifying stress and baseline states using a Support Vector Machine (SVM).

*How it works?*

EEG Data -> 32-channel EEG recordings -> Extracts and calculates Alpha/Beta ratio & Spectral Entropy -> Uses self-reported SAM scores to label stress/baseline -> Trains SVM with RBF kernel -> Generates stress classification results -> Compares predictions with self-reported stress levels

*Why SVM?*

Support Vector Machine is a supervised ML algorithm that can be used to classify data into different classes. An RBF kernel is used to capture non-linear patterns in the extracted EEG features.

*Features Extracted*

1. Alpha/Beta Power Ratio
2. Spectral Entropy
3. Alpha Power
4. Beta Power

*Results*

The model achieved an **AUC** of **0.819** for stress classification. The predicted SVM scores also showed a **Spearman correlation** of **0.489** with the self-reported SAM stress scores.

Dataset: SAM40 - EEG recordings from 40 subjects performing arithmetic, Stroop colour-word and symmetry judgement tasks, along with a relaxed baseline condition.
