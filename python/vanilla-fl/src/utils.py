import numpy as np
import logging
import json
from torch.utils.data import Dataset

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


def sample_noniid(num_clients: int,
                  num_shards: int,
                  num_imgs: int,
                  shards_per_client: int,
                  dataset: Dataset) -> dict:
    """
    Sample non-I.I.D client data from the MNIST dataset using PyTorch's torchvision.datasets.

    :param num_clients: Number of clients/users to sample data for.
    :param num_shards: Number of shards (groups) to divide the dataset into.
    :param num_imgs: Number of images per shard.
    :param shards_per_client: Number of shards assigned to each client.
    :param dataset: Dataset object from which to sample non-I.I.D data.

    :return: Dictionary mapping client IDs to their data indices.
    """
    logging.info("Loading data from dataset object.")

    # Initialize variables
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_clients)}
    total_imgs = num_shards * num_imgs

    # Ensure that the dataset has enough data
    if total_imgs > len(dataset):
        raise ValueError(f"Dataset does not have enough samples. Required: {total_imgs}, Available: {len(dataset)}")

    idxs = np.arange(total_imgs)

    # Extract labels from the dataset
    if hasattr(dataset, 'targets'):
        labels = dataset.targets.numpy()  # For torchvision.datasets.MNIST
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels.numpy()  # Some other datasets might use 'labels'
    else:
        raise ValueError("The dataset does not have a recognized attribute for labels")

    # Align indices with their corresponding labels
    idxs_labels = np.vstack((idxs, labels[:total_imgs]))
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