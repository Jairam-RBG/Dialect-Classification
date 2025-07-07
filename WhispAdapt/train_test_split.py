# -*- coding: utf-8 -*-
"""

@author: r_jairam
"""

import os
import random
import shutil

def split_dataset(source_dir, test_dir, train_dir, valid_dir, split_ratio=(0.70, 0.20, 0.10)):
    for label_folder in os.listdir(source_dir):
        label_source_dir = os.path.join(source_dir, label_folder)
        if not os.path.isdir(label_source_dir):
            continue

        label_train_dir = os.path.join(train_dir, label_folder)
        label_test_dir = os.path.join(test_dir, label_folder)
        label_valid_dir = os.path.join(valid_dir, label_folder)

        os.makedirs(label_train_dir, exist_ok=True)
        os.makedirs(label_test_dir, exist_ok=True)
        os.makedirs(label_valid_dir, exist_ok=True)

        files = os.listdir(label_source_dir)
        random.shuffle(files)

        num_files = len(files)
        train_size = int(num_files * split_ratio[0])
        test_size = int(num_files * split_ratio[1])

        train_files = files[:train_size]
        test_files = files[train_size:train_size + test_size]
        valid_files = files[train_size + test_size:]

        for file in train_files:
            shutil.move(os.path.join(label_source_dir, file), os.path.join(label_train_dir, file))
        for file in test_files:
            shutil.move(os.path.join(label_source_dir, file), os.path.join(label_test_dir, file))
        for file in valid_files:
            shutil.move(os.path.join(label_source_dir, file), os.path.join(label_valid_dir, file))

if __name__ == "__main__":
    parent_folder = '/Source_Directory'
    train_dir = '/train'
    test_dir = '/test'
    valid_dir = '/valid'

    split_dataset(parent_folder, train_dir, test_dir, valid_dir)
