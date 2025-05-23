import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import os
import json
from torchsummary import summary
import random
import matplotlib.pyplot as plt

random.seed(42)

TRAIN = True

base_folder = "../"

models_folder = base_folder + "/models/"
plot_data = base_folder + "/plot_data/"
image_paths_file = base_folder + "/image_paths.txt"

train_paths_file = '../train_paths.json'
val_paths_file = '../val_paths.json'
test_paths_file = '../test_paths.json'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train_history_file = f'{plot_data}train_history.json'
val_history_file = f'{plot_data}val_history.json'
latest_model_file = f'{models_folder}latest_model.pth'
best_model_file = f'{models_folder}best_model.pth'

if not os.path.exists(models_folder):
    os.makedirs(models_folder)

if not os.path.exists(plot_data):
    os.makedirs(plot_data)

"""# **Helper Functions:**"""


def draw_bar_plot(labels):
    unique_labels = list(set(labels))
    counts = [labels.count(label) for label in unique_labels]

    plt.bar(range(len(unique_labels)), counts, color='skyblue')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title('Bar Plot of Labels')
    plt.xticks(range(len(unique_labels)), unique_labels)  # Set x-axis labels
    plt.show()


def list_files(folder_path):
    return [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]


def plotTrainingHistory(train_history, val_history):
    plt.subplot(2, 1, 1)
    plt.title('Cross Entropy Loss')
    plt.plot(train_history['loss'], label='train')
    plt.plot(val_history['loss'], label='val')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.title('Classification Accuracy')
    plt.plot(train_history['accuracy'], label='train')
    plt.plot(val_history['accuracy'], label='val')

    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


def save_dict_to_file(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def get_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def get_saved_dict(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def get_labels(img_paths):
    return list(map(lambda x: int(x[x.rfind('_') + 1:x.rfind('.')]), img_paths))


def get_files():
    image_paths = []
    if os.path.exists(image_paths_file):
        with open(image_paths_file, "r") as file:
            lines = file.readlines()
            # Strip newline characters and append to list
            image_paths = [line.strip() for line in lines]
    else:
        image_paths.extend(list_files(f"{base_folder}/drive_dataset/"))
        for chunk_id in range(1, 7 + 1):
            print(chunk_id)
            image_paths.extend(list_files(f"{base_folder}/generated_dataset/chunk_{chunk_id}"))
        with open(image_paths_file, "w") as file:
            file.writelines(path + "\n" for path in image_paths)
    return image_paths


"""# **Dataset:**"""


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


batch_size = 32
num_workers = 0
image_size = (520, 390)
train_size = 0.7
validation_size = 0.2
test_size = 0.1

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x) if torch.rand(1) < 0.5 else x),
    # Add Gaussian noise to the image
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv5 = nn.Conv2d(256, 64, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 15 * 11, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x.squeeze(1)


def summarize():
    model = CustomCNN()
    if torch.cuda.is_available():
        model = model.cuda()

    # Summarize the model
    summary(model, input_size=(3, *image_size))


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

            pred = torch.clamp(pred, min=1.0)

            # Compute loss between prediction and ground-truth
            loss = loss_fn(pred, y.float())  # Convert labels to float for regression task

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

    return total_loss / num_batches, mean_absolute_error(labels, preds)


def load_model(device, best_model_file):
    model = CustomCNN().to(device)
    checkpoint = torch.load(best_model_file, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    return model


def notebook_driver_code():
    """# **Driver code:**"""

    model = CustomCNN()
    model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    epoch = 0
    train_history = {'loss': [], 'accuracy': []}
    val_history = {'loss': [], 'accuracy': []}
    best_val_loss = float('inf')
    switch_treshold = 1

    if os.path.exists(train_history_file):
        os.remove(train_history_file)
    if os.path.exists(val_history_file):
        os.remove(val_history_file)
    if os.path.exists(latest_model_file):
        os.remove(latest_model_file)

    if TRAIN:
        print("Start training...")
        for t in range(epoch, num_epochs):
            print(f"\nEpoch {t}")

            # Train model for one iteration on training data
            train_loss, train_acc = epoch_iter(train_dataloader, model, criterion, optimizer)
            print(f"Train loss: {train_loss:.3f} \t Train acc: {train_acc:.3f}")

            # Evaluate model on validation data
            val_loss, val_acc = epoch_iter(valid_dataloader, model, criterion, None, is_train=False)
            print(f"Val loss: {val_loss:.3f} \t Val acc: {val_acc:.3f}")

            if train_loss < switch_treshold:
                print("Switched to MAE: " + str(val_loss))
                criterion = nn.L1Loss()

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

        plotTrainingHistory(train_history, val_history)
        print("Finished")

    """Test the model"""

    model = load_model(device, best_model_file)

    test_loss, test_acc = epoch_iter(test_dataloader, model, criterion, is_train=False)
    print(f'\nTest Loss: {test_loss:.3f} \nTest Accuracy: {test_acc:.3f}')


def predict_images(model, image_files, batch_size=32):
    """
    Predicts the output of the given pre-trained PyTorch model for a list of image files in batches.

    Parameters:
    - model: the pre-trained PyTorch model
    - image_files: a list of image file paths
    - batch_size: the number of images to process in a single batch

    Returns:
    - A dictionary where the key is the filename and the value is the output of the model for that key
    """
    model.eval()

    # Define the transformation to be applied to each image
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    results = {}
    num_images = len(image_files)
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_tensors = []

            for image_file in batch_files:
                image = Image.open(image_file).convert('RGB')
                input_tensor = transform(image)
                batch_tensors.append(input_tensor)

            # Stack the tensors to create a batch
            input_batch = torch.stack(batch_tensors)

            outputs = model(input_batch)

            for j, image_file in enumerate(batch_files):
                tmp = outputs[j].item()
                tmp = 1 if tmp < 1 else int(tmp)
                results[image_file] = tmp
    return results


def main():
    notebook_driver_code()
    return
    images = get_json_file(test_paths_file)
    predictions = predict_images(load_model(device, best_model_file), images)
    save_dict_to_file(predictions, 'results.json')


main()
