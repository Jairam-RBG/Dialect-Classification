"""
Auhtor: jairam_r
"""
import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

def process_audio_files_parallel(main_folder, batch_size=50, save_dir='features/', model_name='microsoft/wavlm-large', process_func=None): 
  ## model names options: facebook/hubert-large-ls960-ft, facebook/wav2vec2-xls-r-300m, facebook/wav2vec2-large-xlsr-53
    from src.feature_extraction import clear_gpu_memory, process_audio_file

    audio_files = []
    for label in os.listdir(main_folder):
        class_dir = os.path.join(main_folder, label)
        if os.path.isdir(class_dir):
            for f in os.listdir(class_dir):
                if f.endswith('.wav'):
                    audio_files.append(os.path.join(class_dir, f))
    
    os.makedirs(save_dir, exist_ok=True)
    processed_batches = set()
    progress_file = os.path.join(save_dir, 'processed_batches.pkl')
    
    if os.path.exists(progress_file):
        with open(progress_file, 'rb') as f:
            processed_batches = pickle.load(f)
    
    num_batches = len(audio_files) // batch_size + 1
    for i in range(num_batches):
        if i in processed_batches:
            continue
        batch_files = audio_files[i * batch_size: (i + 1) * batch_size]
        features_dict = {}
        for file in batch_files:
            mean, std, combined = process_func(file, model_name)
            label = os.path.basename(os.path.dirname(file))
            features_dict[file] = {'label': label, 'mean': mean, 'std': std, 'combined': combined}
        with open(f"{save_dir}/features_batch_{i}.pkl", 'wb') as f:
            pickle.dump(features_dict, f)
        processed_batches.add(i)
        with open(progress_file, 'wb') as f:
            pickle.dump(processed_batches, f)
        print(f"Processed batch {i + 1}/{num_batches}")
        clear_gpu_memory()
    return num_batches

def load_and_concatenate_batches(num_batches, save_dir='features/'):
    features_dict = {}
    for i in range(num_batches):
        with open(f"{save_dir}/features_batch_{i}.pkl", 'rb') as f:
            batch = pickle.load(f)
            features_dict.update(batch)
    return features_dict

def prepare_dataset(features_dict):
    X, y = [], []
    for v in features_dict.values():
        X.append(v['mean'])  # Use mean or combined
        y.append(v['label'])
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return np.array(X), np.array(y_encoded), len(le.classes_)
