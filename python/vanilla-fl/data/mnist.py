import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os
from tqdm import tqdm

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Define the directory to save files
save_dir = 'data/'

# Ensure the directory exists; create it if it doesn't
os.makedirs(save_dir, exist_ok=True)

# Function to save dataset with tqdm progress bar
def save_dataset_with_progress(filename, images, labels):
    with tqdm(total=len(images), desc=f'Saving {filename}') as pbar:
        np.savez(os.path.join(save_dir, filename), images=images, labels=labels)
        pbar.update(len(images))

# Save the datasets to npz files in the data directory with progress bar
save_dataset_with_progress('mnist_train.npz', train_images, train_labels)
save_dataset_with_progress('mnist_test.npz', test_images, test_labels)

print(f"Saved MNIST train and test datasets to {save_dir}")
