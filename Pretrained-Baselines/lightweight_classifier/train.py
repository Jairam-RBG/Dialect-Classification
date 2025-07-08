"""
Author: jairam_r
"""
from lightweight_classifier.data_utils import load_data_splits, load_label_mapping
from lightweight_classifier.preprocessing import preprocess_dataset
from lightweight_classifier.trainer_utils import get_model_and_trainer, evaluate_model

import numpy as np
from transformers import AutoFeatureExtractor

# Fix deprecated NumPy type
np.bool = np.bool_

# === CONFIG ===
LANGUAGE = "Tamil"
LABEL_YAML_PATH = "label_mapping.yaml"
DATASET_DIR = f"path"
MODEL_NAME = "name"   ## 'facebook/wav2vec2-xls-r-300m", 'microsoft/wavlm-large', 'facebook/wav2vec2-xls-r-53', 'facebook/hubert-large-ls960-ft'
OUTPUT_DIR = f"{LANGUAGE.lower()}-{MODEL_NAME.split('/')[-1]}"
SAMPLING_RATE = 16000

def main():
    label_mapping = load_label_mapping(LABEL_YAML_PATH, LANGUAGE)
    dataset, label_class = load_data_splits(DATASET_DIR, label_mapping, sampling_rate=SAMPLING_RATE)

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    dataset = preprocess_dataset(dataset, feature_extractor, sampling_rate=SAMPLING_RATE)

    trainer, model = get_model_and_trainer(MODEL_NAME, dataset, label_class.names, output_dir=OUTPUT_DIR)
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)

    evaluate_model(trainer, dataset, label_class.names)

if __name__ == "__main__":
    main()
