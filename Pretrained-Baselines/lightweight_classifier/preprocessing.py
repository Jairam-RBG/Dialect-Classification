"""
Author: jairam_r
"""

def preprocess_dataset(dataset, feature_extractor, sampling_rate=16000):
    """
    Applies the feature extractor to all splits in the dataset.
    """
    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        return feature_extractor(audio_arrays, sampling_rate=sampling_rate, max_length=sampling_rate, truncation=True)

    return dataset.map(preprocess_function, batched=True, remove_columns=["audio"])
