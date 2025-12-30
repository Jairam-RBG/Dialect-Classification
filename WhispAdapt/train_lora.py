# -*- coding: utf-8 -*-
"""
Training WhispAdapt model using LoRA (Rank-32) for Dialect Classification
Author: jairam_r
"""

import os
import yaml
import logging
import evaluate
import numpy as np
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    AutoProcessor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path="config_lora.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)


def load_dataset_and_labels(config):
    logger.info(" Loading dataset...")
    dataset = load_dataset(config["dataset_name"], token=config["hf_token"])
    labels = config["labels"]
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}
    return dataset, label2id, id2label


def load_model_and_processor(config, label2id, id2label):
    logger.info(" Loading processor and model...")
    processor = AutoProcessor.from_pretrained(config["model_name"], token=config["hf_token"])

    model = AutoModelForAudioClassification.from_pretrained(
        config["model_name"],
        quantization_config=config["quantization_config"],
        device_map="auto",
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=["q_proj", "v_proj"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, processor


def get_training_arguments(config):
    return TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        num_train_epochs=config["epochs"],
        evaluation_strategy="steps",
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        save_strategy="steps",
        save_total_limit=2,
        logging_dir=os.path.join(config["output_dir"], "logs"),
        logging_strategy="epoch",
        fp16=True,
        per_device_eval_batch_size=config["batch_size"],
        remove_unused_columns=False,
        label_names=["label"]
    )


def train_model():
    config = load_config()

    login(token=config["hf_token"])

    dataset, label2id, id2label = load_dataset_and_labels(config)
    model, processor = load_model_and_processor(config, label2id, id2label)
    training_args = get_training_arguments(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["test"],
        eval_dataset=dataset["valid"],
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    logger.info(" Starting training...")
    model.config.use_cache = False
    trainer.train()


if __name__ == "__main__":
    train_model()
