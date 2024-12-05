import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)

        # Define pooling and batch normalization layers
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(num_features=32)

        # Define fully connected layers
        self.fc1 = nn.Linear(in_features=256 * 9, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=1)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Dropout
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


class CNN2DModel(nn.Module):
    def __init__(self):
        super(CNN2DModel, self).__init__()

        # Define the convolutional layers
        self.conv_2d_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 1), padding="same")
        self.conv_2d_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1), padding="same")
        self.conv_2d_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1), padding="same")
        self.conv_2d_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 1), padding="same")
        self.conv_2d_5 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 3), padding="same")
        self.conv_2d_6 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), padding="same")
        self.conv_2d_7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), padding="same")
        self.conv_2d_8 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), padding="same")

        # Define pooling and batch normalization layers
        self.pool_2d_1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.pool_2d_2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.bn_2d_1 = nn.BatchNorm2d(num_features=32)
        self.bn_2d_2 = nn.BatchNorm2d(num_features=32)

        self.flatten = nn.Flatten()

        # Define fully connected layers
        self.fc_1 = nn.Linear(in_features=256 * 5 * 17, out_features=700)
        self.fc_2 = nn.Linear(in_features=700, out_features=283)

        # Dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv_2d_1(x))
        x = self.pool_2d_1(x)
        x = self.bn_2d_1(x)

        x = F.relu(self.conv_2d_2(x))
        x = self.pool_2d_1(x)

        x = F.relu(self.conv_2d_3(x))
        x = self.pool_2d_1(x)

        x = F.relu(self.conv_2d_4(x))

        x = F.relu(self.conv_2d_5(x))
        x = self.pool_2d_2(x)
        x = self.bn_2d_2(x)

        x = F.relu(self.conv_2d_6(x))
        x = self.pool_2d_2(x)

        x = F.relu(self.conv_2d_7(x))
        x = self.pool_2d_2(x)

        x = F.relu(self.conv_2d_8(x))
        x = self.pool_2d_2(x)

        x = self.flatten(x)

        x = F.relu(self.fc_1(x))
        x = self.dropout(x)

        x = self.fc_2(x)

        return x
