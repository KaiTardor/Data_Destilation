import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
  def __init__(self, num_classes=10):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(16 * 4 * 4, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, num_classes)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = x.view(x.size(0), -1)  # Aplanar la salida
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
  
class LeNet2(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet2, self).__init__()
        # Acepta 3 canales (RGB)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # FC dims: 16 canales * 4x4 spatial (28->24->12->8->4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # x: [batch, 3, 28, 28]
        x = F.relu(self.conv1(x))      # -> [batch,6,24,24]
        x = self.pool(x)               # -> [batch,6,12,12]
        x = F.relu(self.conv2(x))      # -> [batch,16,8,8]
        x = self.pool(x)               # -> [batch,16,4,4]
        x = x.view(x.size(0), -1)      # -> [batch, 256]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x