import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicRecNet(nn.Module):
    def __init__(self):
        super(MusicRecNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input: (1, 128, 128)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: (32, 64, 64)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: (64, 32, 32)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: (128, 16, 16)

        self.dropout = nn.Dropout(0.3)

        self.flatten_dim = 128 * 16 * 16  # adjust based on input image size
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 128)  # Feature vector
        self.fc3 = nn.Linear(128, 10)   # Assuming 10 genres (GTZAN)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, self.flatten_dim)
        x = self.dropout(F.relu(self.fc1(x)))
        features = self.dropout(F.relu(self.fc2(x)))  # Feature vector
        out = self.fc3(features)
        return out, features  # useful for feeding into traditional classifiers
