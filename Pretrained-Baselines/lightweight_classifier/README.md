# Tamil Dialect Classification (Lightweight Classifier)

This repository provides codes of fair comparison of our WhispAdapt pipeline for dialect classification with the existing state-of-the-art pre-trained models such as

- WavLM
- HuBERT
- XLS-R-53
- XLS-R-300m

All baseline models were fine-tuned using a standardized training configuration aligned with the proposed WhispAdapt framework. 
Each pre-trained self-supervised model was extended with a lightweight classification module integrated atop the final hidden representation of the encoder. 
This classification head consists of two fully connected layers that transform the encoder’s output into dialect-specific class probabilities, similar to the one
used in the proposed architecture.
## Structure

- `lightweight_classifier/`: Utility modules for data, preprocessing, training.
- `train.py`: Main training pipeline.

## Usage

1. Run training:
    ```
    python train.py
    ```

