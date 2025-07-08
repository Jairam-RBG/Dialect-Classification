# Dialect-Classification

###  WhispAdapt: A Few-Shot Fine-Tuning Framework for Multilingual Dialect Classification in Low-Resource Dravidian Languages

This repository contains the codebase for **WhispAdapt**, a framework designed for dialect classification in Dravidian languages. It builds on a pretrained Whisper encoder, which serves as a robust feature extractor and is adapted using Parameter-Efficient Fine-Tuning (PEFT) techniques such as LoRA and QLoRA. Instead of fine-tuning the entire model, only a small number of trainable parameters are introduced through lightweight adapters, enabling efficient adaptation to new tasks. A lightweight classification head is placed on top of the Whisper encoder to perform dialect prediction.

---


## 1 Proposed Architecture — `WhispAdapt/`

`WhispAdapt` is our custom framework that utilizes the **Whisper encoder** as a frozen feature extractor. It is enhanced with **Parameter-Efficient Fine-Tuning (PEFT)** techniques such as:

- 🔹 **LoRA (Low-Rank Adaptation)**
- 🔹 **QLoRA (Quantized LoRA)**

This approach minimizes the number of trainable parameters, making it well-suited for low-resource scenarios while preserving Whisper’s powerful representations.

📂 See [`WhispAdapt/`](./WhispAdapt/) for full implementation, training code, and evaluation scripts.

---

## 2️ Pretrained Baseline Models — `Pretrained_Baselines/`

This section contains **two approaches** using powerful speech encoders like **HuBERT**, **WavLM**, and **XLS-R**:

- `lightweight_classifier/`  
  Uses Hugging Face’s `AutoModelForAudioClassification` for end-to-end classification.
  
- `dnn_classifier/`  
  Uses frozen pretrained models as **feature extractors**, followed by a **custom Deep Neural Network (DNN)** classifier.

 All models are evaluated across multiple dialects and languages to establish baseline performance for comparison with WhispAdapt.

📂 See [`Pretrained_Baselines/`](./Pretrained_Baselines/) for training scripts, config files, and results.

---

## 3️ Traditional Models — `Traditional_Models/`

Includes classical machine learning pipelines using:
-  MFCC + Δ + ΔΔ feature extraction
-  Classifiers like:
  - Support Vector Machines (SVM)
  - Logistic Regression (LR)
  - Deep Neural Networks (DNN)
  - 1D-CNN
  - 2D-CNN
  - LSTM

These serve as simple yet effective baselines to compare against modern Transformer-based approaches.

📂 See [`Traditional_Models/`](./Traditional_Models/) for full code and results.

---

## 🌍 Languages and Dialects

All models are evaluated on a curated dialectal speech dataset for:

| Language   | Dialects                             |
|------------|---------------------------------------|
| **Tamil**     | Madras Vazhaku (Cheannai), Kongu Tamil (Coimbatore), Madurai Vazhaku, Thoothukudi Tamil |
| **Kannada**   | Dakshina Kannada (Coastal), Dharwad (North), Udupi (Kundapura), Mandya     |
| **Malayalam** | Kottayam, Kozhikode, Trivandrum, Thrissur  |

---

## 📊 Evaluation Metrics

All experiments are evaluated using:
- **Accuracy**
- **Precision, Recall, F1-score**
- **Confusion Matrix**
- **Macro-Averaged Scores**
