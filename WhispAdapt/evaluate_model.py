# -*- coding: utf-8 -*-
"""
Evaluate fine-tuned LoRA model and plot confusion matrix
Author: jairam_r
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset
from transformers import AutoProcessor
from peft import PeftModel
import torch
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def evaluate():
    config = load_config()
    dataset = load_dataset(config["dataset_name"], token=config["hf_token"])

    processor = AutoProcessor.from_pretrained(config["model_name"])
    model = PeftModel.from_pretrained(config["output_dir"])
    model.eval()

    trainer_preds = torch.load(f"{config['output_dir']}/trainer_state.json", map_location="cpu")
    trainer = trainer_preds.get("trainer", None)

    if trainer is None:
        from transformers import Trainer
        from transformers import TrainingArguments
        args = TrainingArguments(output_dir=config["output_dir"])
        trainer = Trainer(model=model, args=args)

    predictions = trainer.predict(dataset["train"]).predictions
    predicted_labels = np.argmax(predictions[1], axis=1)

    print(classification_report(dataset["train"]["label"], predicted_labels))

    cm = confusion_matrix(dataset["train"]["label"], predicted_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='turbo')
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate()
