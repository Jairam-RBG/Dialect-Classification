# Pre-trained SOTA model comparison with our proposed approach

This repository contains code for extracting features from dialectal speech data using pre-trained state-of-the-art self-supervised speech models, including Wav2Vec2-XLS-R-53, Wav2Vec2-XLS-R-300m, WavLM, and HuBERT. Features are extracted from the final layer of each model and aggregated using mean and standard deviation pooling to produce 1024-dimensional utterance-level embeddings.

These embeddings are used to train a lightweight deep neural network (DNN) classifier composed of five fully connected layers with progressively reduced dimensions, ReLU activations, and dropout (p = 0.1) to prevent overfitting. The dataset is split into 20% training, 70% testing, and 10% validation. Training is performed for 50 epochs using the Adam optimizer with a learning rate of 1e-4 and a batch size of 100.

The repository includes scripts for feature extraction, dataset preparation, model training, and evaluation.
