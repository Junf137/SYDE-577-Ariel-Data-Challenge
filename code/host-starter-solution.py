# %% Import libraries
import os
import random
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from torchinfo import summary

from cnn_training import CNNModel, CNN2DModel

from training import train, load_checkpoint

from attention_encoder import EncoderDecoderWithAttention, Custom2DDataset

# %% Setup Paths and Read Data
# path to the folder containing the data
data_folder = "../dataset/binned-data/"

# path to the folder containing the train targets and wavelengths information
auxiliary_folder = "../dataset/ariel-data-challenge-2024/"

# output folder
output_dir = "../output"

SEED = 42

do_the_mcdropout_wc = True
do_the_mcdropout = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 673  # total number of observations


# Create a random split of the data
def split(N_total, N_train):
    list_planets = random.sample(range(0, N_total), N_train)
    list_index = np.zeros(N_total, dtype=bool)

    for planet in list_planets:
        list_index[planet] = True

    return list_index


# Validation and train data split (training : validation = 8 : 2)
list_index_train = split(N, 8 * N // 10)

# we have previously cut the data along the wavelengths to remove the edges, this is to match with the targets range in the make data file
cut_inf, cut_sup = (39, 321)
l = cut_sup - cut_inf + 1
wls = np.arange(l)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory {output_dir} created.")
else:
    print(f"Directory {output_dir} already exists.")

# load the data
data_train_AIRS = np.load(f"{data_folder}/data_train.npy")
data_train_FGS = np.load(f"{data_folder}/data_train_FGS.npy")
# data_train_AIRS.shape: (673, 187, 282, 32)
# data_train_FGS.shape: (673, 187, 32, 32)

# solution data
train_solution = np.loadtxt(f"{auxiliary_folder}/train_labels.csv", delimiter=",", skiprows=1)
# train_solution.shape: (673, 284)

# targets:
# exclude the first column from the train_solution, because that column represents the baseline or overall flux
# which might not be needed when predicting the detailed wavelength-dependent features.
#
# The remaining length of 283 matches the length of one training sample label.
targets = train_solution[:, 1:]
# targets.shape: (673, 283)

# targets_mean:
# The mean of the values in each wavelengths of `targets`, excluding the first column (FGS1)
# Used for the 1D-CNN to extract the mean value, only AIRS wavelengths as the FGS point is not used in the white curve
targets_mean = targets[:, 1:].mean(axis=1)
# targets_mean.shape: (673,)

data_train_AIRS_sum3 = data_train_AIRS.sum(axis=3)
# data_train_AIRS_sum3.shape: (673, 187, 282)

# mean of the white curve for each observation
wc_mean = data_train_AIRS_sum3.mean(axis=1).mean(axis=1)
# wc_mean.shape: (673,)

# normalize the white curve
white_curve = data_train_AIRS_sum3.sum(axis=2)
white_curve = white_curve / wc_mean[:, np.newaxis]


def normalize(train, valid):
    # normalise the training and validation data by scaling it into the range [0, 1]
    max_train = train.max()
    min_train = train.min()
    train_norm = (train - min_train) / (max_train - min_train)
    valid_norm = (valid - min_train) / (max_train - min_train)

    return train_norm, valid_norm, min_train, max_train


def unnormalize(data, min_data, max_data):
    return data * (max_data - min_data) + min_data


# Split the light curves and targets
train_wc, valid_wc = white_curve[list_index_train], white_curve[~list_index_train]
train_targets_wc, valid_targets_wc = targets_mean[list_index_train], targets_mean[~list_index_train]

# Normalize the white curve
train_wc_norm, valid_wc_norm, _, _ = normalize(train_wc, valid_wc)

# Normalize the targets
train_targets_wc_norm, valid_targets_wc_norm, min_train_targets_wc, max_train_targets_wc = normalize(
    train_targets_wc, valid_targets_wc
)

# Creating DataLoader for wc training and validation data
# define model hyperparameters
num_epochs = 1200
batch_size = 16


model = CNNModel().to(device)


optimizer = optim.SGD(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=200, gamma=0.2)


loss_fn = torch.nn.MSELoss()


best_valid_loss = float("inf")

# Create TensorDataset for training and validation data
train_dataset = TensorDataset(
    torch.tensor(train_wc_norm[:, np.newaxis, :], dtype=torch.float32),
    torch.tensor(train_targets_wc_norm[:, np.newaxis], dtype=torch.float32),
)
valid_dataset = TensorDataset(
    torch.tensor(valid_wc_norm[:, np.newaxis, :], dtype=torch.float32),
    torch.tensor(valid_targets_wc_norm[:, np.newaxis], dtype=torch.float32),
)

# Create DataLoader for training and validation data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

summary(model, input_size=(train_loader.dataset.tensors[0].shape))

# %% Train the 1D CNN model
LoadOrTrain_1D = "Train"
# LoadOrTrain_1D = "Load"

if LoadOrTrain_1D == "Train":
    train_losses, valid_losses = train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=num_epochs,
        criterion=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        device=device,
        best_valid_loss=best_valid_loss,
    )

    # Plot the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), valid_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir + "/training_validation_loss.png")
    plt.show()

elif LoadOrTrain_1D == "Load":
    model, optimizer, epoch, best_valid_loss = load_checkpoint(output_dir, model, optimizer)

# %% 1D CNN Inference
nb_dropout_wc = 1000


def MC_dropout_WC(model, data_loader, nb_dropout, device):
    """
    Monte Carlo dropout is used to estimate the uncertainty of the model's predictions
    by performing multiple forward passes with dropout enabled.

    """
    model.eval()
    all_predictions = []

    for _ in range(nb_dropout):
        predictions = []

        with torch.no_grad():  # Disable gradient computation
            for x_valid, _ in data_loader:

                x_valid = x_valid.to(device)

                # Forward pass
                output = model(x_valid)
                predictions.append(output.cpu().numpy().flatten())
                # length of predictions: 135 (number of validation samples)

        # Stack predictions for this dropout run and append to all_predictions
        all_predictions.append(np.concatenate(predictions, axis=0))

    return np.array(all_predictions)


if do_the_mcdropout_wc:
    print("Running ...")

    prediction_valid_wc = MC_dropout_WC(model, valid_loader, nb_dropout_wc, device)
    spectre_valid_wc_all = unnormalize(prediction_valid_wc, min_train_targets_wc, max_train_targets_wc)

    spectre_valid_wc = spectre_valid_wc_all.mean(axis=0)
    spectre_valid_std_wc = spectre_valid_wc_all.std(axis=0)

    print("Done.")

    print("prediction_valid_wc.shape", prediction_valid_wc.shape)
    print("spectre_valid_wc_all.shape", spectre_valid_wc_all.shape)
    print("spectre_valid_wc.shape", spectre_valid_wc.shape)
    print("spectre_valid_std_wc.shape", spectre_valid_std_wc.shape)

residuals = spectre_valid_wc - valid_targets_wc

residuals = valid_targets_wc - spectre_valid_wc
print("MSE : ", np.sqrt((residuals**2).mean()) * 1e6, "ppm")

# Calculate the mean absolute percentage error
print(np.mean(100 * np.abs(residuals / valid_targets_wc)))

# np.save(f"{output_dir}/pred_valid_wc.npy", spectre_valid_wc)
# np.save(f"{output_dir}/targ_valid_wc.npy", valid_targets_wc)
# np.save(f"{output_dir}/std_valid_wc.npy", spectre_valid_std_wc)


# %% 2D CNN for atmospheric features

FGS_column = data_train_FGS.sum(axis=2)
# FGS_column.shape: (673, 187, 32)

# concatenate the FGS column to the AIRS data, then squash the the pixels dimension
dataset = np.concatenate([data_train_AIRS, FGS_column[:, :, np.newaxis, :]], axis=2).sum(axis=3)
# dataset.shape: (673, 187, 283)


def norm_star_spectrum(signal):
    # This function assumes that the fist and last 50 time step bins belong to the out of transit
    #   1. Calculate the sum of the mean of the first and last 50 time step bins
    #   2. Normalize the all time step
    img_star = signal[:, :50].mean(axis=1) + signal[:, -50:].mean(axis=1)
    return signal / img_star[:, np.newaxis, :]


# dataset.shape: (673, 187, 283)
dataset_norm = norm_star_spectrum(dataset)
# dataset_norm.shape: (673, 283, 187)


def suppress_mean(targets, mean):
    """
    Suppress the mean of the targets along the columns.
    """
    res = targets - np.repeat(mean.reshape((mean.shape[0], 1)), repeats=targets.shape[1], axis=1)
    return res


train_targets, valid_targets = targets[list_index_train], targets[~list_index_train]
# train_targets.shape (538, 283)
# valid_targets.shape (135, 283)

train_targets_shift = suppress_mean(train_targets, targets_mean[list_index_train])
valid_targets_shift = suppress_mean(valid_targets, targets_mean[~list_index_train])
# train_targets_shift.shape (538, 283)
# valid_targets_shift.shape (135, 283)


# normalization of the targets
def normalize_2(train, valid):
    """
    Normalize the targets by scaling them into the range [-1, 1].

    Note, normalizing both training and validation targets by the same factor from the training set.
    """
    data_min = train.min()
    data_max = train.max()
    train_abs_max = np.max(np.abs([data_min, data_max]))
    train = train / train_abs_max
    valid = valid / train_abs_max
    return train, valid, train_abs_max


train_targets_norm, valid_targets_norm, targets_abs_max = normalize_2(train_targets_shift, valid_targets_shift)

train_obs, valid_obs = dataset_norm[list_index_train], dataset_norm[~list_index_train]


# Subtracting the out transit signal
def suppress_out_transit(data, ingress, egress):
    data_in = data[:, ingress:egress, :]
    return data_in


ingress, egress = 75, 115
train_obs_in = suppress_out_transit(train_obs, ingress, egress)
valid_obs_in = suppress_out_transit(valid_obs, ingress, egress)


# Subtract the mean
def subtract_data_mean(data):
    data_mean = np.zeros(data.shape)
    for i in range(data.shape[0]):
        data_mean[i] = data[i] - data[i].mean()
    return data_mean


train_obs_2d_mean = subtract_data_mean(train_obs_in)
valid_obs_2d_mean = subtract_data_mean(valid_obs_in)

train_obs_norm, valid_obs_norm, train_abs_max = normalize_2(train_obs_2d_mean, valid_obs_2d_mean)

num_epochs_2D = 200
batch_size_2D = 32

model_2D = EncoderDecoderWithAttention(output_dim=283).to(device)

criterion_2D = nn.MSELoss()

optimizer_2D = optim.Adam(model_2D.parameters(), lr=0.001)

best_valid_loss_2D = float("inf")

# preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()
preprocess_2D = transforms.Compose(
    [
        transforms.Resize(232),
        transforms.CenterCrop(224),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ]
)

# Dataset and Dataloader
train_dataset_2D = Custom2DDataset(
    torch.tensor(train_obs_norm[:, np.newaxis, :, :], dtype=torch.float32),
    torch.tensor(train_targets_norm, dtype=torch.float32),
    transform=preprocess_2D,
)
valid_dataset_2D = Custom2DDataset(
    torch.tensor(valid_obs_norm[:, np.newaxis, :, :], dtype=torch.float32),
    torch.tensor(valid_targets_norm, dtype=torch.float32),
    transform=preprocess_2D,
)

train_loader_2D = DataLoader(train_dataset_2D, batch_size=batch_size_2D, shuffle=True)
valid_loader_2D = DataLoader(valid_dataset_2D, batch_size=batch_size_2D, shuffle=False)

summary(model_2D, input_size=(train_loader_2D.dataset.data.shape))

# %% Train the 2D CNN model
TrainOrLoad_2D = "Train"
# TrainOrLoad_2D = "Load"

if TrainOrLoad_2D == "Train":
    train_losses_2D, valid_losses_2D = train(
        model=model_2D,
        train_loader=train_loader_2D,
        valid_loader=valid_loader_2D,
        num_epochs=num_epochs_2D,
        criterion=criterion_2D,
        optimizer=optimizer_2D,
        scheduler=None,
        output_dir=output_dir + "/2D",
        device=device,
        best_valid_loss=best_valid_loss_2D,
    )

    # Plot the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs_2D + 1), train_losses_2D, label="Training Loss 2D")
    plt.plot(range(1, num_epochs_2D + 1), valid_losses_2D, label="Validation Loss 2D")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss (2D)")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_dir + "/2D" + "/training_validation_loss.png")
    plt.show()

elif TrainOrLoad_2D == "Load":
    model_2D, optimizer_2D, num_epochs_2D, best_valid_loss_2D = load_checkpoint(output_dir + "/2D", model_2D, optimizer_2D)

# %% 2D CNN Inference
nb_dropout = 5


def NN_uncertainty(model, x_test, targets_abs_max, nb_dropout, device):
    model.eval()
    all_predictions = []

    for _ in range(nb_dropout):

        batch_predictions = []
        with torch.no_grad():
            for x_valid, _ in x_test:

                pred_norm = model(x_valid.to(device))
                pred = pred_norm * targets_abs_max

                batch_predictions.append(pred.cpu().numpy())

        all_predictions.append(np.concatenate(batch_predictions, axis=0))

    all_predictions = np.array(all_predictions)

    mean = np.mean(all_predictions, axis=0)
    std = np.std(all_predictions, axis=0)

    return mean, std


if do_the_mcdropout:
    spectre_valid_shift, spectre_valid_shift_std = NN_uncertainty(model_2D, valid_loader_2D, targets_abs_max, nb_dropout, device)

residuals = valid_targets_shift - spectre_valid_shift
print("MSE : ", np.sqrt((residuals**2).mean()) * 1e6, "ppm")

# np.save(f'{output_dir}/pred_valid_shift.npy', spectre_valid_shift)
# np.save(f'{output_dir}/targ_valid_shift.npy', valid_targets_shift)
# np.save(f'{output_dir}/std_valid_shift.npy', spectre_valid_shift_std)


# %% Combine 1D and 2D CNN output for FINAL SPECTRA
# ADD THE FLUCTUATIONS TO THE MEAN
predictions_valid = spectre_valid_shift + spectre_valid_wc[:, np.newaxis]

predictions_std_valid = np.sqrt(spectre_valid_std_wc[:, np.newaxis] ** 2 + spectre_valid_shift_std**2)

# final spectra and residuals
predictions = predictions_valid
targets_plot = valid_targets
std = predictions_std_valid

predictions_concatenated_plot = np.concatenate(predictions, axis=0)
wls_concatenated = np.arange(predictions_concatenated_plot.shape[0])
targets_concatenated_plot = np.concatenate(targets_plot, axis=0)
spectre_valid_std_concatenated = np.concatenate(std, axis=0)
residuals = targets_concatenated_plot - predictions_concatenated_plot
uncertainty = spectre_valid_std_concatenated
print("MSE : ", np.sqrt((residuals**2).mean()) * 1e6, "ppm")

# np.save(f'{output_dir}/pred_valid.npy', predictions_valid)
# np.save(f'{output_dir}/std_valid.npy', predictions_std_valid)

# %%
