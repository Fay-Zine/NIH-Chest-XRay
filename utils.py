import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt
from torchvision.models import resnet18 as torchvision_resnet18
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import seaborn as sns
from torchviz import make_dot
from sklearn.metrics import precision_score, recall_score, f1_score

def get_xy(set):

    # load parquet files
    xdf = pd.read_parquet(f'data/X_{set}.parquet')
    ydf = pd.read_parquet(f'data/y_{set}.parquet')

    lengths = xdf['images'].apply(len)
    # get index of lengths that are greater than 50176
    lengths[lengths > 50176].index
    print('Index of images with length greater than 50176: ', lengths[lengths > 50176].index)

    # drop rows with length greater than 50176
    xdf.drop(lengths[lengths > 50176].index, inplace=True)

    # drop corresponding rows in y_val
    ydf.drop(lengths[lengths > 50176].index, inplace=True)

    # convert X to numy array
    X = np.array(xdf['images'].tolist(), dtype='float32')
    X = np.array(X.tolist(), dtype='float32')
    X = X.reshape(-1, 1, 224, 224)

    # convert y to numpy array
    category_to_ordinal = {
    'Pneumonia': 0,
    'Edema': 1,
    'Both': 2
    }
    # Apply the mapping to the 'Target' column
    ydf['Target'] = ydf['Target'].map(category_to_ordinal)

    # Convert y to one-hot encoding
    num_classes = len(category_to_ordinal)
    y = np.eye(num_classes)[ydf['Target']]
    
    return X, y

def load_data():
    # Load the dataset
    X_train, y_train = get_xy('train')
    X_val, y_val = get_xy('val')

    # Convert to PyTorch tensors
    X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
    X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    return train_dataset, val_dataset

def get_dataloaders(train_dataset, val_dataset, batch_size):
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

def load_test_data():
    # Load the dataset
    X_test, y_test = get_xy('test')

    # Convert to PyTorch tensors
    X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)

    # Create datasets
    test_dataset = TensorDataset(X_test, y_test)

    return test_dataset

def compute_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            labels = labels.float()
            outputs = model(inputs)

            # Get the predicted class
            _, predicted = torch.max(outputs, 1)

            # Convert one-hot encoded labels back to class indices for comparison
            _, labels_indices = torch.max(labels, 1)

            total += labels.size(0)
            correct += (predicted == labels_indices).sum().item()

    accuracy = 100 * correct / total
    return accuracy

class EarlyStopping:
    """Early stops training if validation loss doesn't improve after given patience."""
    
    def __init__(self, patience=3, verbose=False, delta=0, config=None,run_name=None):
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
            self.save_checkpoint(val_loss, train_loss, val_accuracy, model, self.config, self.run_name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, train_loss, val_accuracy, model, self.config, self.run_name)
            self.counter = 0

    def get_best_metrics(self):
        return {
            "Best Train Loss": self.best_train_loss,
            "Best Validation Loss": self.val_loss_min,
            "Best Validation Accuracy": self.best_val_accuracy
        }

    def save_checkpoint(self, val_loss, train_loss, val_accuracy, model, config, run_name):
        """Saves model when validation loss decrease."""
        filename=config['model']
        path=f'models/{run_name}_{filename}.pt'
        if self.verbose:
            print(f' Validation loss decreased ({self.val_loss_min:.2e} --> {val_loss:.2e}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
        self.best_train_loss = train_loss
        self.best_val_accuracy = val_accuracy

# Function to process and predict labels
def process_images(model, loader):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0]
            outputs = model(inputs)

            # Apply softmax to convert to probabilities
            probs = F.softmax(outputs, dim=1)
            probabilities.extend(probs.numpy())
    return probabilities

# function to initialize the model
def initialize_CNN(config):

    # Retrieve the model class based on the name in the YAML file
    model_name = config['model']
    model_class = globals().get(model_name)

    # Check if the model class exists
    if model_class is None:
        raise ValueError(f"Model '{model_name}' not found.")

    # Create the model
    model = model_class(config)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    return model, criterion, optimizer

class heavynet(nn.Module):
    """Inspired from lab 9 IFT6390."""

    def __init__(self, config, num_classes=3):
        super().__init__()

        kernel_size = config['kernel_size']
        stride_size = config['stride_size']
        dropout_rate = config['dropout']

        #block1
        self.conv1 = nn.Conv2d(1,32,kernel_size,stride= stride_size, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,32,kernel_size,stride= stride_size, padding = 1)
        self.bn2 = nn.BatchNorm2d(32)

        #block2
        self.conv3 = nn.Conv2d(32,64,kernel_size, stride= stride_size, padding = 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,64,kernel_size, stride= stride_size, padding = 1)
        self.bn4 = nn.BatchNorm2d(64)

        # Calculate the output size after the last convolutional layer
        self.output_size = self._get_conv_output((1, 1, 224, 224))

        #fully connected layer
        self.fc1 = nn.Linear(self.output_size,512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512,num_classes)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(shape)
            output = F.max_pool2d(self.conv2(F.relu(self.bn1(self.conv1(input)))), 2)
            output = F.max_pool2d(self.conv4(F.relu(self.bn3(self.conv3(output)))), 2)
            return output.view(output.size(0), -1).size(1)
            
    def forward(self, x):
      x = F.relu(self.bn1(self.conv1(x)))
      x = F.relu(self.bn2(self.conv2(x)))
      x = F.max_pool2d(x,2)
      x = F.relu(self.bn3(self.conv3(x)))
      x = F.relu(self.bn4(self.conv4(x)))
      x = F.max_pool2d(x,2)

      x = x.view(x.size(0),-1)
      x = F.relu(self.bn5(self.fc1(x)))
      x = self.dropout(x)
      x = self.fc2(x)
      return x

def bootstrap_auc(y_true, y_score, n_bootstraps=100, curve='roc'):
    bootstrapped_scores = []
    rng = np.random.RandomState(42)

    for i in range(n_bootstraps):
        # bootstrap 
        indices = rng.randint(0, len(y_score), len(y_score))
        if len(np.unique(y_true[indices])) < 2:
            continue  # Skip if not valid

        if curve == 'roc':
            # ROC
            fpr, tpr, _ = roc_curve(y_true[indices], y_score[indices])
            score = auc(fpr, tpr)
        elif curve == 'prc':
            # PRC
            precision, recall, _ = precision_recall_curve(y_true[indices], y_score[indices])
            score = auc(recall, precision)

        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # 95% confidence interval
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return confidence_lower, confidence_upper

def plot_roc_curves(y_true, y_pred, class_names):
    # Binarize the output labels for one-vs-rest
    y_true_binarized = label_binarize(y_true, classes=np.arange(len(class_names)))

    # Compute ROC curve and ROC area
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(10, 10))
    colors = ['blue', 'red', 'green'] 

    for i, color in zip(range(len(class_names)), colors):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Calculate confidence intervals
        lower, upper = bootstrap_auc(y_true_binarized[:, i], y_pred[:, i],curve='roc')

        # Plot ROC curve
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='{0}: {1:0.2f} [{2:0.2f}-{3:0.2f}]'.format(class_names[i], roc_auc[i], lower, upper))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    sns.set(style="white", context="talk")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificité')
    plt.ylabel('Sensitivité')
    plt.title('Courbes ROC par classe')
    plt.legend(loc="lower right")
    plt.axis('equal')
    plt.show()


def plot_precision_recall_curves(y_true, y_pred_probs, class_names):
    # Binarize the labels
    y_true_binarized = label_binarize(y_true, classes=np.arange(len(class_names)))

    # Plot the Precision-Recall curve for each class
    plt.figure(figsize=(7, 7))
    for i, color in zip(range(len(class_names)), ['blue', 'red', 'green']):
        precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], y_pred_probs[:, i])
        average_precision = average_precision_score(y_true_binarized[:, i], y_pred_probs[:, i])
        prevalence = np.mean(y_true_binarized[:, i])

        # Calculate confidence intervals
        lower, upper = bootstrap_auc(y_true_binarized[:, i], y_pred_probs[:, i], curve='prc')

        plt.plot(recall, precision, color=color, lw=2,
                 label='{0}: {1:0.2f} [{2:0.2f}-{3:0.2f}]'.format(class_names[i], average_precision, lower, upper))
        plt.axhline(y=prevalence, color=color, linestyle='--', lw=2)

    plt.xlim([0.0, 1.0])
    sns.set(style="white", context="talk")
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (sensitivité)')
    plt.ylabel('Precision (valeur prédictive positive)')
    plt.title('Courbe PRC par classe')
    plt.legend(loc="best")
    plt.axis('equal')
    plt.show()

def bootstrap_brier_score(y_true, y_pred_probs, n_bootstraps=1000):
    bootstrapped_scores = []
    rng = np.random.RandomState(42)  # for reproducibility

    for _ in range(n_bootstraps):
        # Sample with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred_probs), len(y_pred_probs))
        score = brier_score_loss(y_true[indices], y_pred_probs[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Compute 95% confidence interval
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return np.mean(sorted_scores), confidence_lower, confidence_upper

def plot_calibration_curves_with_bootstrap(y_true, y_pred_probs, class_names, n_bootstraps=100, n_bins=10):
    y_true_binarized = label_binarize(y_true, classes=np.arange(len(class_names)))

    plt.figure(figsize=(10, 10))
    colors = ['blue', 'red', 'green']

    for i, color in zip(range(len(class_names)), colors):
        prob_true, prob_pred = calibration_curve(y_true_binarized[:, i], y_pred_probs[:, i], n_bins=n_bins)

        # Calculate Brier score with confidence interval
        brier_score, lower, upper = bootstrap_brier_score(y_true_binarized[:, i], y_pred_probs[:, i])

        # Bootstrapping for error bars
        bootstrapped_prob_true = []

        for j in range(n_bootstraps):
            # Random sampling with replacement
            indices = resample(np.arange(len(y_pred_probs[:, i])), n_samples=len(y_pred_probs[:, i]))
            if len(np.unique(y_true_binarized[indices, i])) < 2:
                # Skip if not enough samples
                continue

            bs_prob_true, _ = calibration_curve(y_true_binarized[indices, i], y_pred_probs[indices, i], n_bins=n_bins)
            bootstrapped_prob_true.append(bs_prob_true)

        # Compute the standard error
        bootstrapped_prob_true = np.array([np.mean(x) for x in zip(*bootstrapped_prob_true)])
        se_prob_true = np.std(bootstrapped_prob_true, axis=0)

        # Linear regression fit
        lr = LinearRegression().fit(prob_pred.reshape(-1, 1), prob_true)
        prob_pred_fit = np.linspace(0, 1, 100)
        prob_true_fit = lr.predict(prob_pred_fit.reshape(-1, 1))

        plt.errorbar(prob_pred, prob_true, yerr=se_prob_true, fmt='o', color=color, label=f'{class_names[i]} (Brier: {brier_score:.2f} [{lower:.2f}-{upper:.2f}])')
        plt.plot(prob_pred_fit, prob_true_fit, color=color, linestyle='-', lw=2)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    sns.set(style="white", context="talk")
    plt.xlabel('Probabilité prédite')
    plt.ylabel('Fraction de classes positives')
    plt.title('Courbes de calibration par classe')
    plt.legend(loc='best')
    plt.axis('equal')
    plt.show()


def visualize_model_architecture(model, input_shape):
    # dummy input tensor
    dummy_input = torch.randn(input_shape)

    # forward pass
    output = model(dummy_input)

    # Create a visualization of the model
    dot = make_dot(output, params=dict(model.named_parameters()))

    # Save or display the visualization
    dot.render('results/network_architecture', format='png')

def calculate_specificity_npv(y_true, y_pred_probs, threshold):
    
    y_pred_thresholded = (y_pred_probs >= threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresholded).ravel()
    
    # Calculate specificity and NPV
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return specificity, npv

def find_optimal_cutoff(y_true, y_pred_probs):
    n_classes = y_pred_probs.shape[1]
    optimal_thresholds = []
    specificities = []
    npvs = []

    for i in range(n_classes):
        # Compute ROC curve for each class
        fpr, tpr, thresholds = roc_curve((y_true == i).astype(int), y_pred_probs[:, i])
        
        # Youden
        J = tpr - fpr
        optimal_idx = np.argmax(J)
        optimal_threshold = thresholds[optimal_idx]

        # Store the results
        optimal_thresholds.append(optimal_threshold)

    return optimal_thresholds


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Vraie classe')
    plt.xlabel('Prediction')
    plt.title('Matrice de confusion')
    plt.show()

def get_predictions_from_probabilities(probs, thresholds):
    pred_labels = np.zeros(probs.shape[0], dtype=int)
    for i in range(probs.shape[1]):
        class_indices = probs[:, i] >= thresholds[i]
        pred_labels[class_indices] = i
    return pred_labels


def calculate_metrics(cm):
    
    metrics_df = pd.DataFrame(columns=['Class', 'PPV', 'NPV', 'Sensitivity', 'Specificity', 'F1 Score'])

    # Calculate the metrics for each class
    for i in range(cm.shape[0]):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        
        # PPV
        PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
        # NPV
        NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
        # Sensitivity
        Sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        # Specificity
        Specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        # F1
        F1 = 2 * (PPV * Sensitivity) / (PPV + Sensitivity) if (PPV + Sensitivity) > 0 else 0
        
        # Append to the DataFrame
        metrics_df.loc[i] = [i, PPV, NPV, Sensitivity, Specificity, F1]
    
    
    return metrics_df