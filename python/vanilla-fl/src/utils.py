import numpy as np
import logging
import json
import torch
from torch import nn
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
                  dataset: Dataset,
                  random_seed: int) -> dict:
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
    
    # random seed
    rng = np.random.default_rng(seed=random_seed)

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
    
    # Shuffle indices
    np.random.shuffle(idxs)

    # Assign shards to each client
    for i in range(num_clients):
        rand_set = set(rng.choice(idx_shard, shards_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    logging.info("Non-I.I.D data sampling completed.")
    return dict_users

def evaluate_model(model, data_loader, device):
    """
    Evaluate a PyTorch model on a given dataset.

    :param model: The trained PyTorch model to evaluate.
    :param data_loader: DataLoader object for the dataset to evaluate on.
    :param device: Device to run the evaluation on ('cpu' or 'cuda').
    :return: Tuple of (accuracy, average_loss) for the dataset.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(data_loader.dataset)
    average_loss = total_loss / len(data_loader)

    logging.info(f'Test Accuracy: {100 * accuracy:.2f}%')
    logging.info(f'Average Test Loss: {average_loss:.6f}')

    return accuracy, average_loss

def aggregate_weights(existing_global_weights, local_weights):
    """
    Aggregate local model weights into global weights by averaging.

    :param existing_global_weights: Dict of global model parameters.
    :param local_weights: List of vectors representing local model parameters from each client.
    :return: Dict of updated global model parameters.
    """
    # Convert existing global weights to a single vector
    global_vector = torch.cat([param.view(-1) for param in existing_global_weights.values()])

    # Initialize the updated global vector with a copy of the global vector
    updated_global_vector = global_vector.clone()

    # Aggregate local vectors into the updated global vector
    for local_vector in local_weights:
        updated_global_vector += local_vector

    # Average the parameters
    updated_global_vector /= (len(local_weights) + 1)

    # Convert the updated global vector back to the original parameter shapes
    pointer = 0
    updated_global_weights = {}
    for name, param in existing_global_weights.items():
        numel = param.numel()  # number of elements in this parameter
        updated_global_weights[name] = updated_global_vector[pointer:pointer + numel].view(param.shape)
        pointer += numel

    return updated_global_weights
