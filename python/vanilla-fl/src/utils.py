import numpy as np
import logging
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def save_dict_to_json(dictionary, json_path):
    """
    Save a dictionary to a JSON file, converting numpy arrays to lists.

    :param dictionary: The dictionary to be saved.
    :param json_path: The path to the JSON file.
    """

    # Helper function to convert numpy arrays to lists
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    logging.info("Starting the conversion of numpy arrays to lists.")
    dictionary = convert(dictionary)
    logging.info("Conversion completed.")

    # Save the converted dictionary to a JSON file
    try:
        with open(json_path, 'w') as f:
            json.dump(dictionary, f)
        logging.info(f"Dictionary successfully saved to {json_path}.")
    except IOError as e:
        logging.error(f"Failed to save dictionary to {json_path}: {e}")


def sample_noniid_new(num_clients: int,
                      num_shards: int,
                      num_imgs: int,
                      shards_per_client: int,
                      train: bool = True,
                      download: bool = True) -> dict:
    """
    Sample non-I.I.D client data from the MNIST dataset using PyTorch's torchvision.datasets.

    :param num_clients: Number of clients/users to sample data for.
    :param num_shards: Number of shards (groups) to divide the dataset into.
    :param num_imgs: Number of images per shard.
    :param shards_per_client: Number of shards assigned to each client.
    :param train: Boolean indicating if training or testing dataset should be used.
    :param download: Boolean indicating if the dataset should be downloaded if not present.

    :return: Dictionary mapping client IDs to their data indices.
    """
    # Load MNIST data
    log_message = "Loading data from torchvision.datasets.MNIST with settings "
    log_message += f"train={train}, download={download}"
    logging.info(log_message)

    transform = transforms.Compose([
        # convert PIL to tensors
        transforms.ToTensor()])
    mnist_dataset = MNIST(root='./data', train=train,
                          download=download, transform=transform)
    # Initialize variables
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards * num_imgs)

    labels = mnist_dataset.targets.numpy()

    # Align indices with their corresponding labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Assign shards to each client
    for i in range(num_clients):
        rand_set = set(np.random.choice(
            idx_shard, shards_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    logging.info("Non-I.I.D data sampling completed.")
    return dict_users


# testing new method
num_clients = 15
num_shards = 200
num_imgs = 60000 // num_shards  # 60000 images in the MNIST training set
shards_per_client = 2

dict_users = sample_noniid_new(
    num_clients=num_clients,
    num_shards=num_shards,
    num_imgs=num_imgs,
    shards_per_client=shards_per_client,
    download=False)

save_dict_to_json(dictionary=dict_users,
                  json_path="client_data.json")


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
        rand_set = set(np.random.choice(
            idx_shard, shards_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    logging.info("Non-I.I.D data sampling completed.")
    return dict_users


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

    # Log the dimensions of the training and testing datasets
    logging.info(
        f"Training dataset dimensions: {[t.shape for t in training_dataset.tensors]}")
    logging.info(
        f"Testing dataset dimensions: {[t.shape for t in testing_dataset.tensors]}")

    return training_dataset, testing_dataset, dict_users
