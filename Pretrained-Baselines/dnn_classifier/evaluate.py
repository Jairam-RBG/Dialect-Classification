"""
Auhtor: jairam_r
"""

import torch
from sklearn.metrics import classification_report

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X).argmax(dim=1)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(all_labels, all_preds))
