import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


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

        # Original Dropout
        # self.dropout1 = nn.Dropout(0.2)
        # self.dropout2 = nn.Dropout(0.1)

        # Increased Dropout Rate
        # self.dropout1 = nn.Dropout(0.5)
        # self.dropout2 = nn.Dropout(0.4)

        # Close to No Dropout
        self.dropout1 = nn.Dropout(0.01)
        self.dropout2 = nn.Dropout(0.01)

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


# Checkpoint saving function
def save_checkpoint(model, valid_loss, best_valid_loss, epoch, optimizer, output_dir):

    if valid_loss < best_valid_loss:

        print(f"Validation loss decreased from {best_valid_loss:.6f} to {valid_loss:.6f}.")

        best_valid_loss = valid_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": valid_loss,
            },
            output_dir + "/model_1d_cnn.pth",
        )

    return best_valid_loss


# Checkpoint loading function
def load_checkpoint(model_dir, model, optimizer):
    model_path = model_dir + "/model_1d_cnn.pth"

    if not os.path.exists(model_path):
        print("Checkpoint file does not exist. Training from scratch.")
        return model, optimizer, 0, float("inf")

    # Load the saved checkpoint
    checkpoint = torch.load(model_path, weights_only=False)

    # Restore model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Retrieve the epoch and loss information
    epoch = checkpoint["epoch"]
    valid_loss = checkpoint["loss"]

    print(f"Checkpoint loaded: Epoch {epoch}, Validation Loss {valid_loss:.4f}")

    return model, optimizer, epoch, valid_loss


# Training Loop
def train(
    model,
    train_loader,
    valid_loader,
    num_epochs,
    criterion,
    optimizer,
    scheduler,
    output_dir,
    device,
    best_valid_loss=float("inf"),
):
    print("Training started.")

    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        # 1. Training phase
        model.train()
        running_loss = 0.0

        train_progress = tqdm(train_loader)
        for x_train, y_train in train_progress:

            x_train, y_train = x_train.to(device), y_train.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(x_train)
            loss = criterion(output, y_train)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_progress.set_description(f"Epoch {epoch+1}/{num_epochs}")
            train_progress.set_postfix({"loss": loss.item()})

        # Learning rate decay
        if scheduler is not None:
            scheduler.step()

        # Average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 2. Validation phase
        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            valid_progress = tqdm(valid_loader)
            for x_valid, y_valid in valid_progress:

                x_valid, y_valid = x_valid.to(device), y_valid.to(device)

                valid_output = model(x_valid)
                valid_loss += criterion(valid_output, y_valid).item()

                valid_progress.set_description(f"Validation Epoch {epoch+1}/{num_epochs}")
                valid_progress.set_postfix({"valid_loss": valid_loss / len(valid_loader)})

        # Average validation loss
        avg_valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)

        # Save the best model checkpoint
        best_valid_loss = save_checkpoint(model, avg_valid_loss, best_valid_loss, epoch, optimizer, output_dir)

    print("Training completed.")

    # Save all train_losses and valid_losses of this training process
    np.save(output_dir + "/train_losses.npy", train_losses)
    np.save(output_dir + "/valid_losses.npy", valid_losses)

    return train_losses, valid_losses
