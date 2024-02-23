# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model class 
class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    # Defines the model forward pass 
    def forward(self, x):
        x = F.relu(self.conv1(x), 2) # chnnels -> 28>26 | rf -> 1>3 | 1>1
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # chnnels -> 26>24>12 | rf -> 3>5>6 | 1>1>2
        x = F.relu(self.conv3(x), 2) # chnnels -> 12>10 | rf -> 6>10 | 2>2
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # CHANNELS -> 10>8>4 | rf -> 10>14>16 | 2>2>4
        x = x.view(-1, 4096) # Output number of nodes -> 4*4*256 = 4096, reshape to 1D tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
