import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from datetime import datetime
import time
import random
from models import CNNMnist
import utils
import json

def main(RANDOM_SEED: int = 42,
         NUM_CLIENTS: int = 3,
         NUM_SHARDS: int = 200,
         NUM_IMGS: int = 300,
         SHARDS_PER_CLIENT: int = 2,
         NUM_ROUNDS: int = 10,
         EPOCHS_PER_CLIENT: int = 10,
         CLIENT_FRACTION: float = 0.5,
         LEARNING_RATE : float = 0.01):
    
    # to calculate execution time
    start_time = time.time()

    # Log input parameters
    logging.info(f"Starting federated learning with the following parameters:")
    logging.info(f"RANDOM_SEED: {RANDOM_SEED}")
    logging.info(f"NUM_CLIENTS: {NUM_CLIENTS}")
    logging.info(f"NUM_SHARDS: {NUM_SHARDS}")
    logging.info(f"NUM_IMGS: {NUM_IMGS}")
    logging.info(f"SHARDS_PER_CLIENT: {SHARDS_PER_CLIENT}")
    logging.info(f"NUM_ROUNDS: {NUM_ROUNDS}")
    logging.info(f"EPOCHS_PER_CLIENT: {EPOCHS_PER_CLIENT}")
    logging.info(f"CLIENT_FRACTION: {CLIENT_FRACTION}")

    # Set seed for reproducibility
    torch.manual_seed(RANDOM_SEED)

    # Load datasets
    logging.info("Loading datasets...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root='./data', 
                        train=True, 
                        download=True, 
                        transform=transform)
    test_dataset = MNIST(root='./data', 
                        train=False, 
                        download=True, 
                        transform=transform)
    logging.info("Datasets loaded successfully.")

    # Split the training dataset into client datasets
    dict_users = utils.sample_noniid(num_clients=NUM_CLIENTS, 
                                    num_imgs=NUM_IMGS, 
                                    num_shards=NUM_SHARDS, 
                                    shards_per_client=SHARDS_PER_CLIENT, 
                                    dataset=train_dataset,
                                    random_seed=RANDOM_SEED)
    logging.info("dict_users generated successfully.")

    # Check if CUDA is available, if not use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Initialize the global model
    global_model = CNNMnist().to(device)
    logging.info("Global model initialized.")
    
    # Training loop for federated learning
    logging.info("Starting federated training...")

    # local losses per round
    round_avg_losses = []

    for round_num in range(NUM_ROUNDS):
        logging.info(f"--- Round {round_num + 1}/{NUM_ROUNDS} ---")
        
        # Copy global weights
        global_weights = global_model.state_dict()

        local_weights = []
        local_weights_copy = []
        local_losses = []

        # Determine the number of clients to sample
        num_clients_sample = max(1, int(NUM_CLIENTS * CLIENT_FRACTION))
        sampled_clients = random.sample(range(NUM_CLIENTS), num_clients_sample)
        sampled_clients.sort()
        logging.info(f"Sampled clients: {sampled_clients}")

        for client_id in sampled_clients:
            local_model = CNNMnist().to(device)
            
            # Broadcasting Global Model
            local_model.load_state_dict(global_weights)

            # Create DataLoader for the client's data subset
            client_indices = dict_users[client_id]
            client_subset = Subset(train_dataset, client_indices)
            client_loader = DataLoader(client_subset, batch_size=64, shuffle=True)

            # Train the local model
            optimizer = torch.optim.Adam(local_model.parameters(), 
                                        lr=LEARNING_RATE, 
                                        weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            local_model.train()
            client_epoch_loss = []

            for epoch in range(EPOCHS_PER_CLIENT):
                for images, labels in client_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    client_epoch_loss.append(loss.item())
                    
            
            avg_local_loss = sum(client_epoch_loss) / len(client_epoch_loss)
            local_losses.append(avg_local_loss)
            
            # collect local weights from the clients
            local_weights.append(local_model.weights_to_vector())
            local_weights_copy.append(local_model.weights_to_vector().tolist())
            
            # logging.info('Local updates collected as a vector.')

            logging.info(f"Client {client_id + 1} - Average Loss: {avg_local_loss:.6f}")
            
        # Calculate average loss for the round
        avg_loss = sum(local_losses) / len(local_losses)
        round_avg_losses.append(avg_loss)
        logging.info(f"Round {round_num + 1} - Average Loss: {avg_loss:.6f}")
        
        # Aggregate the local weights to update the global model
        global_weights = utils.aggregate_weights(existing_global_weights=global_model.state_dict(), 
                                       local_weights=local_weights)
        # logging.info('local updates aggregated by global model.')
        global_model.load_state_dict(global_weights)
        logging.info('Global weights updated.')
        print("")
        
    #  # Write local updates to a JSON file
    # with open(f'local_updates.json', 'w') as f:
    #     json.dump(local_weights_copy, f)
    # # logging.info(f"Local updates for round {round_num + 1} written to JSON file.")

    logging.info("Federated training completed.")
    
    
    
    # Calculate total execution time
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total execution time: {total_time:.2f} seconds")

    # Plot the training loss
    plt.figure()
    plt.plot(range(NUM_ROUNDS), round_avg_losses, marker='o')
    plt.xlabel('Rounds')
    plt.ylabel('Average Loss')
    plt.title('Training Loss over Rounds')
    plt.grid(True)
    plt.savefig('./save/nn_mnist/CNNMnist/federated_{}.png'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    logging.info("Training loss plot saved.")

    # Evaluate the global model
    logging.info("Evaluating the global model on the test dataset...")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    test_accuracy, test_loss = utils.evaluate_model(global_model, test_loader, device)
    logging.info(f"Test on {len(test_dataset)} samples")
    logging.info(f"Test Accuracy: {100 * test_accuracy:.2f}%")
    logging.info(f"Test Loss: {test_loss:.6f}")

if __name__ == '__main__':
    main(NUM_CLIENTS=100,
         CLIENT_FRACTION=0.1)