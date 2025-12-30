"""
Author: jairam_r
"""
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()

    if best_state:
        model.load_state_dict(best_state)
    return model
