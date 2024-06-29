import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from models import CNNMnist
import utils
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Constants
    NUM_CLIENTS = 3
    NUM_SHARDS = 200
    NUM_IMGS = 300
    SHARDS_PER_CLIENT = 2
    # Load datasets from .npz files
    logging.info("Loading datasets...")
    # Define the transformation
    transform = transforms.Compose([transforms.ToTensor()])

    # Load MNIST dataset
    train_dataset = MNIST(root='./data', 
                          train = True, 
                          download = True, 
                          transform = transform)
    test_dataset = MNIST(root = './data', 
                         train = False, 
                         download = True, 
                         transform = transform)
    logging.info("Datasets loaded successfully.")
    
    dict_users = utils.sample_noniid(num_clients = NUM_CLIENTS,
                                     num_imgs = NUM_IMGS,
                                     num_shards = NUM_SHARDS,
                                     shards_per_client = SHARDS_PER_CLIENT,
                                     dataset = train_dataset)
    logging.info("dict_users generated successfully.")
    
    
    # # Check if CUDA is available, if not use CPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logging.info(f'Using device: {device}')

    # # Initialize the CNN model for MNIST
    # global_model = CNNMnist()
    # logging.info("Model initialized.")

    # # Send the model to the selected device (GPU or CPU)
    # global_model.to(device)
    # global_model.train()  # Set the model to training mode
    # logging.info(f"Model architecture:\n{global_model}")
    
    # # Create DataLoaders for training and testing datasets
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # logging.info("DataLoaders created.")

    # # Define optimizer and loss function
    # optimizer = torch.optim.Adam(global_model.parameters(), lr=0.01, weight_decay=1e-4)
    # criterion = nn.CrossEntropyLoss()
    # epochs = 5
    # epoch_loss = []

    # # Training loop
    # logging.info("Starting training...")
    # for epoch in tqdm(range(epochs), desc="Epochs"):
    #     global_model.train()  # Ensure model is in training mode
    #     batch_loss = []

    #     for batch_idx, (images, labels) in enumerate(train_loader):
    #         images, labels = images.to(device), labels.to(device)
    #         logging.info(f"Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")

    #         # Zero gradients, perform a backward pass, and update weights
    #         optimizer.zero_grad()
    #         outputs = global_model(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         if batch_idx % 50 == 0:
    #             logging.info(
    #                 f"Train Epoch: {epoch+1} [{batch_idx * len(images)}/{len(train_loader.dataset)} "
    #                 f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
    #             )
    #         batch_loss.append(loss.item())

    #     # Calculate average loss for the epoch
    #     loss_avg = sum(batch_loss) / len(batch_loss)
    #     logging.info(f"Epoch {epoch+1} - Average Loss: {loss_avg:.6f}")
    #     epoch_loss.append(loss_avg)

    # logging.info("Training completed.")

    # # Plot loss
    # plt.figure()
    # plt.plot(range(len(epoch_loss)), epoch_loss, marker='o')
    # plt.xlabel('Epochs')
    # plt.ylabel('Train Loss')
    # plt.title('Training Loss over Epochs')
    # plt.grid(True)
    # plt.savefig('../save/nn_mnist_CNNMnist_{}.png'.format(epochs))
    # logging.info("Training loss plot saved.")

    # # test_acc, test_loss = utils.test_inference(global_model, test_loader, device)
    # # logging.info(f'Test on {len(test_dataset)} samples')
    # # logging.info(f"Test Accuracy: {100*test_acc:.2f}%")

if __name__ == '__main__':
    main()
