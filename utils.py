import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import copy

dataset_paths = {
    3: (
        "/data/train_images_disease.parquet",
        "/data/val_images_disease.parquet",
        "/data/test_images_disease.parquet",
        "/data/train_df_disease.parquet",
    ),
    5: (
        "/data/train_images.parquet",
        "/data/val_images.parquet",
        "/data/test_images.parquet",
        "/data/train_df.parquet",
    ),
}


class ParquetImageDataset(Dataset):
    def __init__(self, parquet_filename, class_to_idx, transform=None):
        self.parquet_table = pq.read_table(parquet_filename).to_pandas()
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.parquet_table)

    def __getitem__(self, idx):
        row = self.parquet_table.iloc[idx]
        image_bytes = np.frombuffer(row["images"], dtype=np.float32)
        image = torch.tensor(image_bytes.reshape(3, 224, 224)).to(torch.float32)
        if self.transform:
            image = self.transform(image)
        label_idx = self.class_to_idx[row["labels"]]
        return image, label_idx


def prepare_dataset(df_path):
    train_df = pd.read_parquet(df_path)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["Target"])
    class_to_idx = {cls: idx for idx, cls in enumerate(label_encoder.classes_)}
    target_counts = train_df["Target"].value_counts()
    num_samples = len(train_df)
    class_weights = {c: num_samples / count for c, count in target_counts.items()}
    return class_to_idx, class_weights


def load_data(num_class):
    if num_class not in dataset_paths:
        raise ValueError(f"Unsupported number of classes: {num_class}")

    paths = dataset_paths[num_class]
    df_path = paths[3]
    class_to_idx, class_weights = prepare_dataset(df_path)

    train_dataset = ParquetImageDataset(paths[0], class_to_idx)
    valid_dataset = ParquetImageDataset(paths[1], class_to_idx)
    test_dataset = ParquetImageDataset(paths[2], class_to_idx)

    return train_dataset, valid_dataset, test_dataset


def get_dataloaders(train_dataset, valid_dataset, test_datasset, batch_size):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_datasset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return train_loader, valid_loader, test_loader


def compute_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


class EarlyStopping:
    """Early stops training if validation loss doesn't improve after given patience."""

    def __init__(self, patience=5, verbose=False, delta=0, config=None, run_name=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.config = config
        self.run_name = run_name

    def __call__(self, val_loss, train_loss, val_accuracy, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(
                val_loss, train_loss, val_accuracy, model, self.config, self.run_name
            )
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(
                val_loss, train_loss, val_accuracy, model, self.config, self.run_name
            )
            self.counter = 0

    def save_checkpoint(
        self, val_loss, train_loss, val_accuracy, model, config, run_name
    ):
        """Saves model when validation loss decrease."""
        filename = config["model"]
        path = f"models/{run_name}_{filename}.pt"
        if self.verbose:
            print(
                f" Validation loss decreased ({self.val_loss_min:.2e} --> {val_loss:.2e}).  Saving model ..."
            )
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

        # Log metrics to wandb
        wandb.log(
            {
                "Train Loss": train_loss,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_accuracy,
            }
        )

        artifact = wandb.Artifact(filename, type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)


# Function to process and predict labels
def process_images(model, loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0]
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.numpy())
    return predictions


# function to initialize the model
class convnet6(nn.Module):
    def __init__(self, params, num_classes=24):
        super(convnet6, self).__init__()

        kernel_size = params["model"]["kernel_size"]
        stride_size = params["model"]["stride_size"]
        dropout_rate = params["model"]["dropout"]
        padding = (kernel_size - 1) // 2
        activation = (
            nn.ReLU() if params["model"]["activation"] == "relu" else nn.Sigmoid()
        )

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=kernel_size, stride=stride_size, padding=padding
            ),
            activation,
            nn.AvgPool2d(2) if params["model"]["pooling"] == "avg" else nn.MaxPool2d(2),
            nn.Dropout(dropout_rate),
            *(
                nn.Sequential(
                    nn.Conv2d(
                        32,
                        32,
                        kernel_size=kernel_size,
                        stride=stride_size,
                        padding=padding,
                    ),
                    activation,
                    nn.Dropout(dropout_rate),
                )
                for _ in range(5)
            ),
        )

        # Calculate the size for the linear layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)
            output_size = self.conv_layers(dummy_input).view(-1).shape[0]

        # Fully connected layer
        self.fc = nn.Linear(output_size, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the linear layer
        return self.fc(x)

    # function to initialize the model
    def initialize_CNN(config):
        # Retrieve the model class based on the name in the YAML file
        model_name = config["model"]
        model_class = globals().get(model_name)

        # Check if the model class exists
        if model_class is None:
            raise ValueError(f"Model '{model_name}' not found.")

        # Create the model
        model = model_class(config)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])

        return model, criterion, optimizer

    # function to plot learning curves
    def plot_learning_curves(train_losses, val_losses, filename, i):
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, "bo-", label="Training Loss")
        plt.plot(epochs, val_losses, "ro-", label="Validation Loss")
        plt.title("Training and Validation Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.ylim(0.9, 1)
        plt.xticks(epochs)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"results/{filename}_{i}.png")


class convnet13(nn.Module):
    def __init__(self, params, num_classes=24):
        super(convnet13, self).__init__()

        kernel_size = params["model"]["kernel_size"]
        stride_size = params["model"]["stride_size"]
        dropout_rate = params["model"]["dropout"]
        padding = (kernel_size - 1) // 2
        activation = (
            nn.ReLU() if params["model"]["activation"] == "relu" else nn.Sigmoid()
        )

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            activation,
            *(
                nn.Sequential(
                    nn.Conv2d(128 if i > 0 else 64, 128, kernel_size=3, padding=1),
                    activation,
                )
                for i in range(5)
            ),
            *(
                nn.Sequential(
                    nn.Conv2d(256 if i > 0 else 128, 256, kernel_size=3, padding=1),
                    activation,
                )
                for i in range(3)
            ),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            activation,
        )

        # Calculate the size for the linear layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)
            output_size = self.conv_layers(dummy_input).view(-1).shape[0]

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(output_size, 256), activation, nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class resnet18(nn.Module):
    def __init__(self, params, num_classes=24):
        super(resnet18, self).__init__()
        self.model = torchvision_resnet18(weights=None)

        # Adjusting the first convolutional layer for 1 channel input
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Adjusting the final fully connected layer for 24 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class heavynet(nn.Module):
    """Inspired from lab 9 IFT6390."""

    def __init__(self, config, num_classes=24):
        super().__init__()

        kernel_size = config["kernel_size"]
        stride_size = config["stride_size"]
        dropout_rate = config["dropout"]

        # block1
        self.conv1 = nn.Conv2d(1, 32, kernel_size, stride=stride_size, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size, stride=stride_size, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # block2
        self.conv3 = nn.Conv2d(32, 64, kernel_size, stride=stride_size, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size, stride=stride_size, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # Calculate the output size after the last convolutional layer
        self.output_size = self._get_conv_output((1, 1, 28, 28))

        # fully connected layer
        self.fc1 = nn.Linear(self.output_size, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(shape)
            output = F.max_pool2d(self.conv2(F.relu(self.bn1(self.conv1(input)))), 2)
            output = F.max_pool2d(self.conv4(F.relu(self.bn3(self.conv3(output)))), 2)
            return output.view(output.size(0), -1).size(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
