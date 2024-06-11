import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import json
import random
import argparse
import torchvision.models as models

random.seed(42)


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


class LegoDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


def parse_args():
    parser = argparse.ArgumentParser(description='Run Task2')
    parser.add_argument('--model_name', type=str, default='custom',
                        choices=['custom', 'resnet', 'vgg', 'densenet', 'efficientnet'])
    parser.add_argument('--path_to_model', type=str, default='models/custom.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_json', type=str, default='test.json')
    parser.add_argument('--output_json', type=str, default='output.json')
    return parser.parse_args()


def get_tarnsformer(model_name):
    if model_name == 'custom':
        return transforms.Compose([
            transforms.Resize((520, 390)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif model_name == 'efficientnet':
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_model(model_name, path_to_model):
    if model_name == 'custom':
        model = CustomCNN()
    elif model_name == 'vgg16':
        model = models.vgg16()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 1)
        model = nn.DataParallel(model)
    elif model_name == 'resnet':
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        model = nn.DataParallel(model)
    elif model_name == 'densenet':
        model = models.densenet201()
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 1)
        model = nn.DataParallel(model)
    elif model_name == 'efficientnet':
        model = models.efficientnet_v2_s()
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 1)
        model = nn.DataParallel(model)
    else:
        raise ValueError('Model not supported')

    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def epoch_iter(dataloader, model, device, model_name):
    # Get number of batches
    model.eval()

    preds = []

    # Enable/disable gradients based on whether the model is in train or evaluation mode
    with torch.set_grad_enabled(False):

        # Analyse all batches
        for batch, X in enumerate(tqdm(dataloader)):
            X = X.to(device)

            # Forward pass to obtain prediction of the model
            pred = model(X)

            preds.extend(pred.detach().cpu().numpy())

    if model_name == 'custom':
        preds_int = list(map(lambda x: max(int(x + 0.5), 1), preds))
    else:
        preds_int = list(map(lambda x: max(int(x[0] + 0.5), 1), preds))

    return preds_int


def get_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path_imgs = get_json_file(args.img_folder)
    dataset = LegoDataset(path_imgs, transform=get_tarnsformer(args.model_name))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model(args.model_name, args.path_to_model)
    model.to(device)

    preds = epoch_iter(dataloader, model, device, args.model_name)

    with open(args.output_json, 'w') as file:
        json.dump(preds, file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
