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
from sklearn.metrics import classification_report, confusion_matrix


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
        print("Loading data...")
        train_dataset, val_dataset = load_data()
        train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, config["batch_size"])

        # initialize the model
        model, criterion, optimizer = initialize_CNN(config)

        # Early stopping
        early_stopping = EarlyStopping(
            patience=3, verbose=True, config=config, run_name=run_name
        )

        # Initialize variables to track the best metrics
        best_metrics = None

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
                labels = labels.float()
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Print status every batch
                if i % 1 == 0:
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
                    labels = labels.float()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            train_loss = running_loss / len(train_loader)
            val_accuracy = compute_accuracy(model, val_loader)

            # Validation Phase for Accuracy
            print(
                f"Epoch [{epoch+1}/{config.num_epochs}], Validation Accuracy: {val_accuracy:.2f}%, Val Loss: {val_loss:.2e}"
            )

            # Early Stopping based on validation loss
            early_stopping(val_loss, train_loss, val_accuracy, model)
            if early_stopping.early_stop:
                print("\nEarly stopping")
                best_metrics = early_stopping.get_best_metrics()
                break

        # If training completes without early stopping, retrieve the best metrics
        if not early_stopping.early_stop:
            best_metrics = early_stopping.get_best_metrics()

        # Log the best metrics to wandb
        if best_metrics:
            wandb.log(best_metrics)


def evaluate_model(run_name):
    # load config file
    with open("params_eval.yaml", "r") as file:
        config = yaml.safe_load(file)["parameters"]
        config = {key: value["values"][0] if key != 'classes' else value["values"] for key, value in config.items()}

    # Load the best model
    print("Loading model...")
    filename = config["model"]
    path = f"models/{run_name}_{filename}.pt"

    # load .pt model
    model, _, _ = initialize_CNN(config)
    model.load_state_dict(torch.load(path))

    test = load_test_data()
    test_loader = DataLoader(test, shuffle=False)

    true_labels =[]

    for _, labels in test_loader:
        true_labels.append(labels)
    true_labels = torch.cat(true_labels, dim=0)

    if len(true_labels.shape) > 1:  # Check if labels are one-hot encoded
        true_labels = torch.max(true_labels, 1)[1]  # Convert to class indices

    print("Evaluating model...")
    pred_probs = process_images(model, test_loader)
    pred_probs = np.array(pred_probs)

    # Plot ROC curves
    print("Plotting curves...")
    plot_roc_curves(true_labels.numpy(), pred_probs, config['classes'])
    plot_precision_recall_curves(true_labels.numpy(), pred_probs, config['classes'])
    plot_calibration_curves_with_bootstrap(true_labels.numpy(), pred_probs, config['classes'])

    # Optimal threshold
    print("Calculating optimal threshold...")
    optimal_thresholds = find_optimal_cutoff(true_labels.numpy(), pred_probs)
    print(f'Optimal thresholds: {optimal_thresholds}')

    # Calculate the predicted labels using the optimal thresholds
    pred_labels_optimal = (pred_probs >= optimal_thresholds).astype(int)

    # Calculate the confusion matrix
    print("Calculating confusion matrix...")
    pred_labels_optimal = get_predictions_from_probabilities(pred_probs, optimal_thresholds)
    cm = confusion_matrix(true_labels.numpy(), pred_labels_optimal)
    plot_confusion_matrix(cm, class_names=config['classes'])

    # Calculate the classification report
    print("Calculating classification report...")
    metrics_df = calculate_metrics(cm)
    metrics_df.to_csv('results/classification_metrics.csv', index=False)

    # model architecture
    print("Visualizing model architecture...")
    visualize_model_architecture(model, input_shape=(1, 1, 224, 224))
    return


####################
# Main hyperparameter tuning loop
####################

# Load config file
with open("params.yaml") as f:
    sweep_config = yaml.safe_load(f)

# Initialize the sweep
print("Initializing sweep")
sweep_id = wandb.sweep(sweep_config, project="MMD6020 NIH Chest X-Ray")

# Start the sweep
print("Starting sweep")
wandb.agent(sweep_id, train)

####################
# Evaluate model
####################

# #load saved model from .pt file
# run_name='stellar-sweep-3'
# evaluate_model(run_name)
