import numpy as np
import logging
import json
import torch
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sample_noniid(npz_path: str, 
                  num_clients: int, 
                  num_shards: int, 
                  num_imgs: int, 
                  shards_per_client: int) -> dict:
    """
    Sample non-I.I.D client data from MNIST dataset stored in .npz file.
    
    :param npz_path: Path to the .npz file containing the MNIST dataset.
    :param num_clients: Number of clients/users to sample data for.
    :param num_shards: Number of shards (groups) to divide the dataset into.
    :param num_imgs: Number of images per shard.
    :param shards_per_client: Number of shards assigned to each client.
    
    :return: Dictionary mapping client IDs to their data indices.
    """

    logging.info(f"Loading data from {npz_path}")
    data = np.load(npz_path)
    labels = data['labels']

    # Initialize variables
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards * num_imgs)

    # Align indices with their corresponding labels 
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Assign shards to each client
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, shards_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    
    logging.info("Non-I.I.D data sampling completed.")
    return dict_users


def test_sample_noniid():
    partitions = sample_noniid(npz_path='data/mnist_train.npz',
                            num_clients=7,
                            num_shards = 200, 
                            num_imgs = 300, 
                            shards_per_client = 2)

    # Convert NumPy arrays to lists for JSON serialization
    for key in partitions:
        partitions[key] = partitions[key].tolist()
        
    # Define the path where you want to save the JSON file
    json_output_path = 'client_data.json'

    # Convert the dictionary to JSON format and write it to a file
    with open(file = json_output_path, 
            mode='w', 
            encoding='utf-8') as json_file:
        json.dump(partitions, json_file)

    print(f"Output stored in {json_output_path}")

def numpy_to_tensor(npz_path: str) -> TensorDataset:
    """
    Convert numpy arrays from .npz file to PyTorch TensorDataset.

    :param npz_path: Path to the .npz file containing images and labels.
    :return: TensorDataset containing images and labels as PyTorch tensors.
    """
    logging.info(f"Loading data from {npz_path}")
    data = np.load(npz_path)
    images_np = data['images']
    labels_np = data['labels']

    # Convert to PyTorch tensors
    images_tensor = torch.tensor(images_np, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_np, dtype=torch.long)

    # Create TensorDataset
    dataset = TensorDataset(images_tensor, labels_tensor)
    
    logging.info("Conversion to PyTorch TensorDataset completed.")
    return dataset

# Test the numpy_to_tensor function
import matplotlib.pyplot as plt
def test_numpy_to_tensor(npz_path):
    print(f"Testing numpy_to_tensor function with file: {npz_path}")
    
    # Call numpy_to_tensor to obtain TensorDataset
    dataset = numpy_to_tensor(npz_path)
    
    # Display first few images with labels using matplotlib
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
    for i, ax in enumerate(axes):
        image, label = dataset[i]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

test_numpy_to_tensor('data/mnist_train.npz')
test_numpy_to_tensor('data/mnist_test.npz')

def get_datasets(npz_train: str, 
                 npz_test: str,
                 num_clients: int, 
                 num_shards: int = 200, 
                 num_imgs: int = 300, 
                 shards_per_client: int = 2) -> tuple:
    """
    Returns train and test datasets and a dictionary of user data.

    :param npz_train: Path to the .npz file containing the training dataset.
    :param npz_test: Path to the .npz file containing the testing dataset.
    :param num_clients: Number of clients/users to sample data for.
    :param num_shards: Number of shards (groups) to divide the dataset into.
    :param num_imgs: Number of images per shard.
    :param shards_per_client: Number of shards assigned to each client.
    
    :return: Tuple containing training dataset, testing dataset, and dictionary of user data.
    """
    logging.info("Sampling non-I.I.D. data for clients.")
    dict_users = sample_noniid(npz_path=npz_train, 
                               num_clients=num_clients,
                               num_shards=num_shards,
                               num_imgs=num_imgs,
                               shards_per_client=shards_per_client)
    
    logging.info("Converting training and testing datasets to TensorDataset.")
    training_dataset = numpy_to_tensor(npz_train)
    testing_dataset = numpy_to_tensor(npz_test)
    
    return training_dataset, testing_dataset, dict_users