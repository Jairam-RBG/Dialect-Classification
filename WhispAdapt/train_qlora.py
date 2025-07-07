'''
Training  WhispAdapt model using QLoRA (Rank-32) for Dialect Classification
Author: jairam_r
'''

import os
import yaml
import torch
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
    BitsAndBytesConfig,
)

from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(path="config_qlora.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(dataset_name: str, token: str):
    logger.info(f" Loading dataset: {dataset_name}")
    return load_dataset(dataset_name, token=token)

def get_label_mappings(labels):
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}
    return label2id, id2label
  
def load_model_and_processor(model_name, labels, label2id, id2label):
    processor = AutoProcessor.from_pretrained(model_name)

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        device_map="auto",
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    model = prepare_model_for_kbit_training(model)
    return model, processor

def apply_lora(model, lora_config_dict):
    lora_cfg = LoraConfig(
        r=lora_config_dict["rank"],
        lora_alpha=lora_config_dict["alpha"],
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_config_dict["dropout"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model

def compute_metrics(eval_pred):
    acc = evaluate.load("accuracy")
    preds = np.argmax(eval_pred.predictions, axis=1)
    return acc.compute(predictions=preds, references=eval_pred.label_ids)

def setup_trainer(model, processor, dataset, training_args_cfg, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_args_cfg["batch_size"],
        gradient_accumulation_steps=training_args_cfg["gradient_accumulation_steps"],
        learning_rate=training_args_cfg["learning_rate"],
        warmup_ratio=training_args_cfg["warmup_ratio"],
        num_train_epochs=training_args_cfg["epochs"],
        evaluation_strategy="steps",
        eval_steps=training_args_cfg["eval_steps"],
        save_steps=training_args_cfg["save_steps"],
        save_strategy="steps",
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="epoch",
        fp16=True,
        per_device_eval_batch_size=training_args_cfg["batch_size"],
        remove_unused_columns=False,
        label_names=["label"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["test"],
        eval_dataset=dataset["valid"],
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    return trainer

def main():
    config = load_config()
    login(token=config["hf_token"])

    dataset = load_data(config["dataset_name"], config["hf_token"])
    label2id, id2label = get_label_mappings(config["labels"])

    model, processor = load_model_and_processor(
        config["model_name"], config["labels"], label2id, id2label
    )
    model = apply_lora(model, config["lora"])

    trainer = setup_trainer(model, processor, dataset, config, config["output_dir"])
    model.config.use_cache = False

    logger.info(" Starting QLoRA fine-tuning...")
    trainer.train()


if __name__ == "__main__":
    main()
