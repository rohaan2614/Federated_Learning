from torch import nn
import torch.nn.functional as F
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class CNNMnist(nn.Module):
    def __init__(self, num_channels = 1, num_classes = 10):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
        logging.debug(f"Initialized CNNMnist with conv1: {self.conv1}, conv2: {self.conv2}, fc1: {self.fc1}, fc2: {self.fc2}")

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def weights_to_vector(self):
        """Returns a single vector with all model weights."""
        return torch.cat([param.view(-1) for param in self.parameters()])

    def vector_to_weights(self, vector):
        """Load a single vector back into model's original weight shapes."""
        pointer = 0
        for param in self.parameters():
            numel = param.numel()  # number of elements in this parameter
            param_shape = param.shape  # shape of the parameter
            param.data = vector[pointer:pointer + numel].view(param_shape)
            pointer += numel
            
    def zero_out_weights(self):
        for param in self.parameters():
            param.data.fill_(0)
