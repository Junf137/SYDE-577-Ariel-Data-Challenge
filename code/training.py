import os
import numpy as np
import torch

from tqdm import tqdm


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
