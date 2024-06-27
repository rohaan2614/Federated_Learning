import numpy as np
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sample_noniid(npz_path : str, num_clients : int, num_shards = 200, num_imgs = 300) -> dict:
    """
    Sample non-I.I.D client data from MNIST dataset stored in .npz file.
    :param npz_path: Path to the .npz file.
    :param num_clients: Number of users/clients.
    :return: Dictionary mapping client IDs to their data indices.
    """
    logging.info(f"Loading data from {npz_path}")
    data = np.load(npz_path)
    images = data['images']
    labels = data['labels']

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards * num_imgs)

    # align the indices with their corresponding labels 
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Assign 2 shards per client
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # update idx_shard: remove the selected shards.
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    
    logging.info("Non-I.I.D data sampling completed.")
    return dict_users


partitions = sample_noniid(npz_path='data/mnist_train.npz',
                           num_clients=15)

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