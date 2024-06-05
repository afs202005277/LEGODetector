# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import torch.nn.functional as F
import os
import json
from torchsummary import summary
import random
import numpy as np
import matplotlib.pyplot as plt
random.seed(42)

base_folder = "../"

models_folder = base_folder + "/models/"
plot_data = base_folder + "/plot_data/"
image_paths_file = base_folder + "/image_paths.txt"

train_paths_file = '../train_paths.json'
val_paths_file = '../val_paths.json'
test_paths_file = '../test_paths.json'

if not os.path.exists(models_folder):
    os.makedirs(models_folder)

if not os.path.exists(plot_data):
    os.makedirs(plot_data)

"""# **Helper Functions:**"""

def plotTrainingHistory(train_history, val_history):
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.title('Mean Square Error (loss)')
    plt.plot(train_history['loss'], label='train', marker='o')
    plt.plot(val_history['loss'], label='val', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.title('Mean Absolute Error (acc)')
    plt.plot(train_history['accuracy'], label='train', marker='o')
    plt.plot(val_history['accuracy'], label='val', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

def get_labels(img_paths):
  return list(map(lambda x: int(x[x.rfind('_')+1:x.rfind('.')]), img_paths))

def save_dict_to_file(data, filename):
  with open(filename, 'w') as file:
    json.dump(data, file)

def get_json_file(filename):
  with open(filename, 'r') as file:
      data = json.load(file)
  return data

"""# **Tranformers**"""

resnet_transform = transforms.Compose([
    transforms.Resize((224, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

efficient_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform = resnet_transform

"""# **Dataset:**"""

batch_size = 32
num_workers = 2
train_size = 0.8
validation_size = 0.1
test_size = 0.1

class LegoDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

train_paths = get_json_file(train_paths_file)
val_paths = get_json_file(val_paths_file)
test_paths = get_json_file(test_paths_file)

train_labels = get_labels(train_paths)
val_labels = get_labels(val_paths)
test_labels = get_labels(test_paths)

train_dataset = LegoDataset(train_paths, train_labels, transform=transform)
valid_dataset = LegoDataset(val_paths, val_labels, transform=transform)
test_dataset = LegoDataset(test_paths, test_labels, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

"""# **Model:**"""

import torchvision.models as models

"""## ResNet50"""

## Load ResNet model from torchvision (with pretrained=True)
#resnet = models.resnet50(pretrained=True)
#
## Disable Gradients
#for param in resnet.parameters():
#    param.requires_grad = False
#
## Change the number of neurons in the last layer to the number of classes of the CIFAR10 dataset
#num_ftrs = resnet.fc.in_features
#resnet.fc = nn.Linear(num_ftrs, 1)

"""## VGG16"""

from torchvision.models import VGG16_Weights

vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

for param in vgg16.parameters():
    param.requires_grad = False

num_ftrs = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_ftrs, 1)

for param in vgg16.classifier[6].parameters():
    param.requires_grad = True

"""## DenseNet"""

#from torchvision.models import DenseNet201_Weights
#
#densenet = models.densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
#
#for param in densenet.parameters():
#    param.requires_grad = False
#
#num_ftrs = densenet.classifier.in_features
#densenet.classifier = nn.Linear(num_ftrs, 1)
#
#for param in densenet.classifier.parameters():
#    param.requires_grad = True

"""## Efficient Net"""

#from torchvision.models import EfficientNet_V2_S_Weights
#
#efficientnet = models.efficientnet_v2_s(EfficientNet_V2_S_Weights.IMAGENET1K_V1)
#
#for param in efficientnet.parameters():
#    param.requires_grad = False
#
#num_ftrs = efficientnet.classifier[1].in_features
#efficientnet.classifier[1] = nn.Linear(num_ftrs, 1)
#
#for param in efficientnet.classifier.parameters():
#    param.requires_grad = True

model = vgg16
best_model_file = models_folder + "vgg16_0001_un_best_model.pth"
latest_model_file = models_folder + "vgg16_0001_un_latest_model.pth"

train_history_file = plot_data + "vgg16_0001_un_train_history.json"
val_history_file = plot_data + "vgg16_0001_un_val_history.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model)
model.to(device)

learning_rate = 0.0001
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

"""# **Train the model:**"""

# Define the training loop
def epoch_iter(dataloader, model, loss_fn, optimizer=None, is_train=True):
    if is_train:
        assert optimizer is not None, "When training, please provide an optimizer."

    # Get number of batches
    num_batches = len(dataloader)

    # Set model to train mode or evaluation mode
    if is_train:
        model.train()
    else:
        model.eval()

    # Define variables to save predictions and labels during the epoch
    total_loss = 0.0
    preds = []
    labels = []

    # Enable/disable gradients based on whether the model is in train or evaluation mode
    with torch.set_grad_enabled(is_train):

        # Analyse all batches
        for batch, (X, y) in enumerate(tqdm(dataloader)):

            # Put data in same device as model (GPU or CPU)
            X, y = X.to(device), y.to(device)

            # Forward pass to obtain prediction of the model
            pred = model(X)

            # Compute loss between prediction and ground-truth
            loss = loss_fn(pred, y.float().unsqueeze(1))  # Convert labels to float for regression task

            # Backward pass
            if is_train:
                # Reset gradients in optimizer
                optimizer.zero_grad()
                # Calculate gradients by backpropagating loss
                loss.backward()
                # Update model weights based on the calculated gradients
                optimizer.step()

            # Save training metrics
            total_loss += loss.item()  # IMPORTANT: call .item() to obtain the value of the loss WITHOUT the computational graph attached

            # Add predictions
            preds.extend(pred.detach().cpu().numpy())
            labels.extend(y.cpu().numpy())

    return total_loss / num_batches, mean_absolute_error(labels, preds), labels, preds

def train(model, num_epochs, train_dataloader, validation_dataloader, loss_fn, optimizer, num_epochs_to_unfreeze = -1, switch_treshold = 1):
    train_history = {'loss': [], 'accuracy': []}
    val_history = {'loss': [], 'accuracy': []}
    best_val_loss = float('inf')
    already_switch = False

    print("Start training...")

    for t in range(0, num_epochs):
        print(f"\nEpoch {t+1}")

        if t + 1 == num_epochs_to_unfreeze:
            for param in model.parameters():
              param.requires_grad = True

            print("Switched to MSE")
            loss_fn = nn.MSELoss()
            already_switch = False

        # Train model for one iteration on training data
        train_loss, train_acc, _a, _b = epoch_iter(train_dataloader, model, loss_fn, optimizer)
        print(f"Train loss: {train_loss:.3f} \t Train acc: {train_acc:.3f}")

        # Evaluate model on validation data
        val_loss, val_acc, _a, _b = epoch_iter(valid_dataloader, model, loss_fn, None, is_train=False)
        print(f"Val loss: {val_loss:.3f} \t Val acc: {val_acc:.3f}")

        if not already_switch and train_loss < switch_treshold:
          print("Switched to MAE: " + str(train_loss))
          loss_fn = nn.L1Loss()
          already_switch = True

        # Save model when validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
            torch.save(save_dict, best_model_file)

        # Save latest model
        save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
        torch.save(save_dict, latest_model_file)

        # Save training history for plotting purposes
        train_history["loss"].append(train_loss)
        train_history["accuracy"].append(train_acc)

        val_history["loss"].append(val_loss)
        val_history["accuracy"].append(val_acc)

        save_dict_to_file(train_history, train_history_file)
        save_dict_to_file(val_history, val_history_file)

    print("Finished")
    return train_history, val_history

num_epochs = 30
num_epochs_to_unfreeze = 10

train_history, val_history = train(model, num_epochs, train_dataloader, valid_dataloader, loss, optimizer, num_epochs_to_unfreeze)

plotTrainingHistory(train_history, val_history)


"""# Test the model"""

from sklearn.metrics import accuracy_score

checkpoint = torch.load(best_model_file)
model.load_state_dict(checkpoint['model'])

test_loss, test_acc, labels, preds = epoch_iter(test_dataloader, model, loss, is_train=False)
print(f'\nMean Square Error: {test_loss:.3f} \nMean Absolute Error: {test_acc:.3f}')
