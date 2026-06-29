import torch
import torch.nn as nn
import torch.optim as optim

class AccentCNN(nn.Module):
    """
    A 2D Convolutional Neural Network for Accent Classification.
    Built according to the St. Petersburg architecture using raw 2D time-frequency features.
    """
    def __init__(self, num_classes, dropout_rate=0.25, input_shape=(64, 75)):
        super(AccentCNN, self).__init__()
        
        # Convolution Block 1: 32 blocks (filters) of size (3,3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        # Pooling layer with filter size (3,3) and a stride of 2
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        
        # Convolution Block 2: 64 blocks (filters) of size (3,3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        
        # Calculate the flattened size dynamically based on the input matrix dimensions
        self.flatten_dim = self._get_flatten_dim(input_shape)
        
        # Dense Layer 1: 128 neurons with ReLU activation
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Dense Layer 2: Output neurons equal to the number of target accents
        self.fc2 = nn.Linear(128, num_classes)
        
        # Softmax activation for the final probability distribution vector
        self.softmax = nn.Softmax(dim=1)

    def _get_flatten_dim(self, input_shape):
        """Helper to calculate the output size of the spatial convolutions when flattened."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
            x = self.pool1(self.conv1(dummy_input))
            x = self.pool2(self.conv2(x))
            return x.numel()

    def forward(self, x):
        # Pass through Block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        # Pass through Block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        # Flatten for Dense layers
        x = x.view(x.size(0), -1)
        
        # Fully Connected Block
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output Classification Layer
        x = self.fc2(x)
        
        # Note: Softmax is explicit here to yield probability vectors as requested, 
        # but if using PyTorch's nn.CrossEntropyLoss during training, feed it raw logits instead.
        probas = self.softmax(x)
        return x, probas


def get_training_configurations(model, lr=0.001):
    """
    Configures the categorical cross-entropy loss and the Adam optimizer 
    with explicit beta coefficients derived from the literature.
    """
    # Categorical Cross Entropy equivalent in PyTorch (expects raw logits)
    criterion = nn.CrossEntropyLoss()
    
    # Adam Optimization with beta1 = 0.9 and beta2 = 0.999
    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr, 
        betas=(0.9, 0.999)
    )
    
    return criterion, optimizer
