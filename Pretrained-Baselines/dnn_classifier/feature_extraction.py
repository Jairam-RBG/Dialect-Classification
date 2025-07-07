"""
Author: jairam_r
"""
import os
import pickle
import numpy as np
import librosa
import torch
import warnings
from transformers import WavLMModel, Wav2Vec2FeatureExtractor # example usage of wavlm model

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def extract_wavlm_features(audio, sr, model_name='microsoft/wavlm-large'):
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name).to(device)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            outputs = model(**inputs)
    return outputs.last_hidden_state.cpu().squeeze(0).numpy()

def pool_features(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return mean, std, np.concatenate((mean, std), axis=0)

def process_audio_file(audio_path, model_name='microsoft/wavlm-large'):
    audio, sr = librosa.load(audio_path, sr=None)
    features = extract_wavlm_features(audio, sr, model_name)
    return pool_features(features)
