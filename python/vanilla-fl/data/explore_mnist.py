import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset from the .npz files
train_data = np.load('data/mnist_train.npz')
test_data = np.load('data/mnist_test.npz')

train_images = train_data['images']
train_labels = train_data['labels']
test_images = test_data['images']
test_labels = test_data['labels']

# Check the shape and type of the data
print(f"Train Images Shape: {train_images.shape}, Type: {train_images.dtype}")
print(f"Train Labels Shape: {train_labels.shape}, Type: {train_labels.dtype}")
print(f"Test Images Shape: {test_images.shape}, Type: {test_images.dtype}")
print(f"Test Labels Shape: {test_labels.shape}, Type: {test_labels.dtype}")

# Function to plot sample images
def plot_sample_images(images, labels, n=10):
    plt.figure(figsize=(10, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.show()

# Plot first 10 images from the training set
plot_sample_images(train_images, train_labels)

# Plot first 10 images from the test set
plot_sample_images(test_images, test_labels)
