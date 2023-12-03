import os

os.environ["PYTHONHASHSEED"] = str(42)

import torch
import time
import yaml
from utils import *
from torch.utils.data import DataLoader
import pandas as pd
import wandb
import random


# Function to train and evaluate the model
def train():
    with wandb.init() as run:
        run_name = run.name  # Unique identifier for the current run
        config = run.config

        # Set random seed and get number of epochs
        torch.manual_seed(config["random_seed"])
        np.random.seed(config["random_seed"])
        random.seed(config["random_seed"])

        num_epochs = config["num_epochs"]

        # Load data
        train_dataset, val_dataset, test_dataset = load_data(
            3
        )  # Specify the number of classes, either 3 or 5
        train_loader, val_loader, test_loader = get_dataloaders(
            train_dataset, val_dataset, test_dataset, config["batch_size"]
        )

        # initialize the model
        model, criterion, optimizer = initialize_CNN(config)

        # Early stopping
        early_stopping = EarlyStopping(
            patience=5, verbose=True, config=config, run_name=run_name
        )

        # Training loop
        print("Training model...")
        # Initialize lists to track training and validation losses
        train_losses, val_losses = [], []
        for epoch in range(config.num_epochs):
            model.train()
            running_loss = 0.0
            start_time = time.time()

            for i, (inputs, labels) in enumerate(train_loader, 1):
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Print status every 10 batches
                if i % 10 == 0:
                    time_elapsed = time.time() - start_time
                    avg_time_per_batch = time_elapsed / i
                    eta_seconds = avg_time_per_batch * (len(train_loader) - i)

                    # Convert ETA to minutes and seconds
                    eta_minutes, eta_seconds = divmod(eta_seconds, 60)

                    print(
                        f"\rEpoch [{epoch+1}/{config.num_epochs}], Batch [{i}/{len(train_loader)}], "
                        f"Loss: {running_loss / i:.4f}, ETA: {int(eta_minutes)}m {int(eta_seconds)}s",
                        end="",
                        flush=True,
                    )

            # Validation Phase for Loss
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            train_loss = running_loss / len(train_loader)

            # Append losses for plotting
            train_losses.append(running_loss / len(train_loader))
            val_losses.append(val_loss)
            val_accuracy = compute_accuracy(model, val_loader)

            # Early Stopping based on validation loss
            early_stopping(val_loss, train_loss, val_accuracy, model)
            if early_stopping.early_stop:
                print("\nEarly stopping")
                break

            # Validation Phase for Accuracy
            print(
                f"Epoch [{epoch+1}/{config.num_epochs}], Validation Accuracy: {val_accuracy:.2f}%, Val Loss: {val_loss:.2e}"
            )
            test_accuracy = compute_accuracy(model, test_loader)
            print(f"Test Accuracy: {test_accuracy:.2f}%")

            # Log metrics to wandb
            wandb.log(
                {
                    "Train Loss": running_loss / len(train_loader),
                    "Validation Loss": val_loss,
                    "Validation Accuracy": val_accuracy,
                    "Test Accuracy": test_accuracy,
                }
            )


def evaluate_model(run_name):
    # load config file
    with open("params_eval.yaml", "r") as file:
        config = yaml.safe_load(file)["parameters"]
        config = {key: value["values"][0] for key, value in config.items()}

    # Load the best model
    filename = config["model"]
    path = f"models/{run_name}_{filename}.pt"

    # load .pt model
    # model = torch.load(path)
    model, _, _ = initialize_CNN(config)
    model.load_state_dict(torch.load(path))

    train, val, test = load_data(3)
    test_loader = DataLoader(test, shuffle=False)
    pred = process_images(model, test_loader)
    report = classification_report(test.Target, pred)
    return report


####################
# Main hyperparameter tuning loop
####################

# Load config file
with open("params.yaml") as f:
    sweep_config = yaml.safe_load(f)

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="Kaggle ASCII Competition")

# Start the sweep
wandb.agent(sweep_id, train)

####################
# Evaluate model
####################

# load saved model from .pt file
# run_name='trim-sweep-20'
# evaluate_model(run_name)
