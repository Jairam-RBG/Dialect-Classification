# -*- coding: utf-8 -*-
"""
Feature extraction using Whisper + Upload to Hugging Face Hub
"""

import os
import json
import yaml
import argparse
import logging
from datasets import Dataset, DatasetDict, Audio, ClassLabel
from transformers import AutoProcessor
from huggingface_hub import login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_label_map(language: str, mapping_path: str) -> dict:
    with open(mapping_path, 'r') as f:
        mappings = json.load(f)
    if language not in mappings:
        raise ValueError(f"Language '{language}' not found in label mapping.")
    return mappings[language]


def create_data_dict(data_dir: str, label_map: dict) -> dict:
    data = {"audio": [], "label": []}
    for label_name in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        label = label_map.get(label_name)
        if label is None:
            logger.warning(f"Label '{label_name}' not found in mapping. Skipping.")
            continue
        for fname in os.listdir(label_path):
            if fname.endswith(".wav"):
                data["audio"].append(os.path.join(label_path, fname))
                data["label"].append(label)
    logger.info(f"Collected {len(data['audio'])} audio files from {data_dir}")
    return data


def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features[0]
    return batch


def main():
    parser = argparse.ArgumentParser(description="Feature extraction and upload pipeline")
    parser.add_argument("--config", type=str, default="config_features.yaml", help="Path to config file")
    parser.add_argument("--mapping", type=str, default="language_mapping.json", help="Path to label mapping JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    language = config["language"]
    label_map = load_label_map(language, args.mapping)

    train_dir = config["paths"]["train"]
    test_dir = config["paths"]["test"]
    valid_dir = config["paths"]["valid"]

    model_id = config["huggingface"]["model_id"]
    repo_id = config["huggingface"]["repo_id"]
    hf_token = config["huggingface"]["token"]

    processor = AutoProcessor.from_pretrained(model_id)

    logger.info(f"Preparing dataset for language: {language}")
    datasets = {
        "train": Dataset.from_dict(create_data_dict(train_dir, label_map)),
        "test": Dataset.from_dict(create_data_dict(test_dir, label_map)),
        "valid": Dataset.from_dict(create_data_dict(valid_dir, label_map)),
    }

    class_label = ClassLabel(num_classes=len(label_map), names=list(label_map.keys()))
    speech = DatasetDict(datasets)
    speech = speech.cast_column("audio", Audio(sampling_rate=16000))
    speech = speech.cast_column("label", class_label)

    logger.info("Extracting features...")
    processed_dataset = speech.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=["audio"],
        num_proc=1,
        desc="Extracting features"
    )

    logger.info("Logging into Hugging Face Hub...")
    login(token=hf_token)

    logger.info(f"Pushing dataset to: {repo_id}")
    processed_dataset.push_to_hub(repo_id, token=hf_token)
    logger.info("Upload complete.")


if __name__ == "__main__":
    main()
