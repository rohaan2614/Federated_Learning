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
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
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
    
    # Check if CUDA is available, if not use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Initialize the CNN model for MNIST
    global_model = CNNMnist()
    logging.info("Model initialized.")

    # Send the model to the selected device (GPU or CPU)
    global_model.to(device)
    global_model.train()  # Set the model to training mode
    logging.info(f"Model architecture:\n{global_model}")
    
    # Create DataLoaders for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    logging.info("DataLoaders created.")

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(global_model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    epochs = 5
    epoch_loss = []

    # Training loop
    logging.info("Starting training...")
    for epoch in tqdm(range(epochs), desc="Epochs"):
        global_model.train()  # Ensure model is in training mode
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients, perform a backward pass, and update weights
            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                logging.info(
                    f"Train Epoch: {epoch+1} [{batch_idx * len(images)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )
            batch_loss.append(loss.item())

        # Calculate average loss for the epoch
        loss_avg = sum(batch_loss) / len(batch_loss)
        logging.info(f"Epoch {epoch+1} - Average Loss: {loss_avg:.6f}")
        epoch_loss.append(loss_avg)

    logging.info("Training completed.")
    
    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    file_name = f'./save/nn_mnist/CNNMnist/'
    file_name += f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{epochs}-epochs.png'
    plt.savefig(file_name)
    logging.info("Training loss plot saved.")

    test_accuracy, test_loss = utils.evaluate_model(model = global_model, 
                                                    data_loader = test_loader, 
                                                    device = device)
    logging.info(f'Test on {len(test_dataset)} samples')
    logging.info(f"Test Accuracy: {100*test_accuracy:.2f}%")
    logging.info(f"Test Loss: {test_loss:.6f}%")

if __name__ == '__main__':
    main()
