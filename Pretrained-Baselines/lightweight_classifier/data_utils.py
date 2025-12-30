"""
Author: jairam_r
"""

import os
import shutil
import random
import yaml
from datasets import Dataset, DatasetDict, Audio, ClassLabel

def load_label_mapping(yaml_path, language):
    """
    Loads label mapping for the specified language from a YAML file.
    """
    with open(yaml_path, 'r') as f:
        all_mappings = yaml.safe_load(f)
    if language not in all_mappings:
        raise ValueError(f"Language '{language}' not found in label mapping YAML.")
    return all_mappings[language]

def split_dataset(source_dir, output_base_dir, split_ratio=(0.7, 0.2, 0.1)):
    """
    Splits dataset into train, test, and validation folders by class labels.
    """
    assert sum(split_ratio) == 1.0, "Split ratio must sum to 1.0"
    
    for label in os.listdir(source_dir):
        src_label_dir = os.path.join(source_dir, label)
        if not os.path.isdir(src_label_dir):
            continue

        files = os.listdir(src_label_dir)
        random.shuffle(files)

        n = len(files)
        train_split = int(n * split_ratio[0])
        test_split = int(n * split_ratio[1])

        partitions = {
            os.path.join(output_base_dir, "train"): files[:train_split],
            os.path.join(output_base_dir, "test"): files[train_split:train_split + test_split],
            os.path.join(output_base_dir, "valid"): files[train_split + test_split:]
        }

        for split_dir, file_list in partitions.items():
            target_dir = os.path.join(split_dir, label)
            os.makedirs(target_dir, exist_ok=True)
            for file in file_list:
                shutil.copy(os.path.join(src_label_dir, file), os.path.join(target_dir, file))

def prepare_dataset(directory, label_mapping):
    """
    Constructs HuggingFace Dataset from audio file paths and labels.
    """
    data_dict = {"audio": [], "label": []}
    for label_name in os.listdir(directory):
        class_dir = os.path.join(directory, label_name)
        if not os.path.isdir(class_dir):
            continue

        label = label_mapping.get(label_name)
        for file_name in os.listdir(class_dir):
            data_dict["audio"].append(os.path.join(class_dir, file_name))
            data_dict["label"].append(label)
    return Dataset.from_dict(data_dict)

def load_data_splits(base_dir, label_mapping, sampling_rate=16000):
    """
    Loads train/test/validation datasets from directory structure.
    """
    train = prepare_dataset(os.path.join(base_dir, "train"), label_mapping)
    test = prepare_dataset(os.path.join(base_dir, "test"), label_mapping)
    valid = prepare_dataset(os.path.join(base_dir, "valid"), label_mapping)

    dataset = DatasetDict({"train": train, "test": test, "valid": valid})
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    labels = ClassLabel(num_classes=len(label_mapping), names=list(label_mapping.keys()))
    dataset = dataset.cast_column("label", labels)

    return dataset, labels
