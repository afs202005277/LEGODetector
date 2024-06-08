import os
import random
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
import torchvision.models as models
from torchvision.models import VGG16_Weights

models_folder = "../models/"
plot_data = "../plot_data/"

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


def save_dict_to_file(data, filename):
  with open(filename, 'w') as file:
    json.dump(data, file)
    
class PixelAccuracyLoss(nn.Module):
    def __init__(self):
        super(PixelAccuracyLoss, self).__init__()

    def forward(self, outputs, targets):
        # Ensure the outputs and targets are of the same shape
        assert outputs.shape == targets.shape, "Shape mismatch between outputs and targets"

        # Count the number of correctly predicted pixels
        correct_pixels = (outputs == targets)
        
        print(outputs.numel() - correct_pixels.sum())

        return outputs.numel() - correct_pixels.sum()
    
"""# **Dataset:**"""

class LegoDataset(Dataset):
    def __init__(self, image_paths, masks, transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.masks = masks
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.masks[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask)

        return image, mask


masks = [f.split('.')[0] for f in os.listdir('../masks') if f.endswith('.jpg')]
images = [f.split('.')[0] for f in os.listdir('../generated_dataset') if f.endswith('.jpg')]
images = list(set(images).intersection(masks))
print(len(images))


train_images = images[:int(len(images)*0.7)]
validation_images = images[int(len(images)*0.7):int(len(images)*0.8)]
test_images = images[int(len(images)*0.8):]

batch_size = 16
num_workers = 0
image_size = (200, 100)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x) if torch.rand(1) < 0.5 else x),  # Add Gaussian noise to the image
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(x.size(0), -1)),
])

train_images_names = [ '../generated_dataset/' + f + '.jpg' for f in train_images]
train_masks_names = [ '../masks/' + f + '.jpg' for f in train_images]
validation_images_names = [ '../generated_dataset/' + f + '.jpg' for f in validation_images]
validation_masks_names = [ '../masks/' + f + '.jpg' for f in validation_images]
test_images_names = [ '../generated_dataset/' + f + '.jpg' for f in test_images]
test_masks_names = [ '../masks/' + f + '.jpg' for f in test_images]

train_dataset = LegoDataset(train_images_names, train_masks_names, transform=transform, mask_transform=mask_transform)
valid_dataset = LegoDataset(validation_images_names, validation_masks_names, transform=transform, mask_transform=mask_transform)
test_dataset = LegoDataset(test_images_names, test_masks_names, transform=transform, mask_transform=mask_transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers, shuffle=False)


vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

for param in vgg16.parameters():
    param.requires_grad = False

num_ftrs = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_ftrs, 200 * 100)

for param in vgg16.classifier[6].parameters():
    param.requires_grad = True
    
best_model_file = models_folder + "vgg16_seg_0001_un_best_model.pth"
latest_model_file = models_folder + "vgg16_seg_0001_un_latest_model.pth"

train_history_file = plot_data + "vgg16_seg_0001_un_train_history.json"
val_history_file = plot_data + "vgg16_seg_0001_un_val_history.json"
    
model = vgg16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

learning_rate = 0.0001
loss_fn = PixelAccuracyLoss()

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
            y = y.squeeze(1)
            # Forward pass to obtain prediction of the model
            pred = model(X)

            # Compute loss between prediction and ground-truth
            loss = loss_fn(pred, y)

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

    print("Start training...")

    for t in range(0, num_epochs):
        print(f"\nEpoch {t+1}")

        if t + 1 == num_epochs_to_unfreeze:
            for param in model.parameters():
              param.requires_grad = True

        # Train model for one iteration on training data
        train_loss, train_acc, _a, _b = epoch_iter(train_dataloader, model, loss_fn, optimizer)
        print(f"Train loss: {train_loss:.3f} \t Train acc: {train_acc:.3f}")

        # Evaluate model on validation data
        val_loss, val_acc, _a, _b = epoch_iter(valid_dataloader, model, loss_fn, None, is_train=False)
        print(f"Val loss: {val_loss:.3f} \t Val acc: {val_acc:.3f}")

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

        #save_dict_to_file(train_history, train_history_file)
        #save_dict_to_file(val_history, val_history_file)

    print("Finished")
    return train_history, val_history

num_epochs = 70
num_epochs_to_unfreeze = 5

train_history, val_history = train(model, num_epochs, train_dataloader, valid_dataloader, loss_fn, optimizer, num_epochs_to_unfreeze)

#plotTrainingHistory(train_history, val_history)

"""Test the model"""

'''vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

for param in vgg16.parameters():
    param.requires_grad = False

num_ftrs = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_ftrs, 200 * 100)

for param in vgg16.classifier[6].parameters():
    param.requires_grad = True

model = vgg16
model.to(device)
checkpoint = torch.load('vgg16_seg_0001_un_best_model.pth')
model.load_state_dict(checkpoint['model'])
criterion = PixelAccuracyLoss()
test_loss, test_acc, _, _ = epoch_iter(test_dataloader, model, criterion, is_train=False)
print(f'\nTest Loss: {test_loss:.3f} \nTest Accuracy: {test_acc:.3f}')

for batch, (X, y) in enumerate(tqdm(test_dataloader)):
    X, y = X.to(device), y.to(device)

    pred = model(X).unsqueeze(1)
    
    reshaped_tensor = pred.view(image_size[0], image_size[1])
    true_mask = y.view(image_size[0], image_size[1])

    # Define the reverse transform
    reverse_transform = transforms.Compose([
        transforms.ToPILImage()
    ])

    # Apply the reverse transform to get the image
    mask_image = reverse_transform(reshaped_tensor)
    true_mask_image = reverse_transform(true_mask)

    mask_image.save('here.png')
    true_mask_image.save('true.png')'''
    

            
