"""
Author: jairam_r
"""

import numpy as np
import evaluate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
)

def compute_metrics(eval_pred):
    """
    Accuracy evaluation metric.
    """
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return evaluate.load("accuracy").compute(predictions=predictions, references=eval_pred.label_ids)

def get_model_and_trainer(model_name, dataset, label_names, output_dir):
    """
    Loads pre-trained model and sets up the HuggingFace Trainer.
    """
    label2id = {label: str(i) for i, label in enumerate(label_names)}
    id2label = {str(i): label for i, label in enumerate(label_names)}

    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        label2id=label2id,
        id2label=id2label
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        warmup_ratio=0.2,
        logging_steps=40,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=None,
        compute_metrics=compute_metrics
    )

    return trainer, model

def evaluate_model(trainer, dataset, label_names):
    """
    Generates classification report and confusion matrix for test split.
    """
    preds_output = trainer.predict(dataset["test"])
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = dataset["test"]["label"]

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=label_names))

    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='turbo', xticklabels=label_names, yticklabels=label_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
