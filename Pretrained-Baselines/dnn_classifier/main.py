"""
Author: jairam_r
"""

from dnn_classifier.feature_extraction import process_audio_file
from dnn_classifier.dataset_preparation import process_audio_files_parallel, load_and_concatenate_batches, prepare_dataset
from dnn_classifier.dnn_model import AudioClassifier
from dnn_classifier.train import train_model
from dnn_classifier.evaluate import evaluate_model

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

# Step 1: Extract features
num_batches = process_audio_files_parallel(
    main_folder='data/',
    batch_size=300,
    save_dir='features/',
    process_func=process_audio_file
)

# Step 2: Prepare dataset
features = load_and_concatenate_batches(num_batches, save_dir='features/')
X, y, num_classes = prepare_dataset(features)

# Step 3: Split data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# Step 4: Dataloaders
train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train)), batch_size=100, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val)), batch_size=100)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test)), batch_size=100)

# Step 5: Train model
model = AudioClassifier(num_classes)
model = train_model(model, train_loader, val_loader, num_epochs=50)
torch.save(model.state_dict(), 'models/WavLM_Tamil_classifier_50.pth')

# Step 6: Evaluate
evaluate_model(model, test_loader)
