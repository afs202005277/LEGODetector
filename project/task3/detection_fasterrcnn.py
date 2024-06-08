import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import transforms
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, average_precision_score
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch_snippets import Report
import torchvision.transforms.functional as F
import torchvision.transforms.transforms as T
import xml.etree.ElementTree as ET
import matplotlib.patches as patches

class Compose:
    """
    Composes several torchvision image transforms 
    as a sequence of transformations.
    Inputs
        transforms: list
            List of torchvision image transformations.
    Returns
        image: tensor
        target: dict
    """
    def __init__(self, transforms = []):
        self.transforms = transforms
    # __call__ sequentially performs the image transformations on
    # the input image, and returns the augmented image.
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
class ToTensor(torch.nn.Module):
    """
    Converts a PIL image into a torch tensor.
    Inputs
        image: PIL Image
        target: dict
    Returns
        image: tensor
        target: dict
    """
    def forward(self, image, target = None):
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image, target
class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """
    Randomly flips an image horizontally.
    Inputs
        image: tensor
        target: dict
    Returns
        image: tensor
        target: dict
    """
    def forward(self, image, target = None):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                width, _ = F.get_image_size(image)
                target["boxes"][:, [0, 2]] = width - \
                                     target["boxes"][:, [2, 0]]
        return image, target
    
root_path = 'datasets/vc/'
images_path = 'datasets/vc/images/'
labels_path = 'datasets/vc/labels/'

def get_object_detection_model(num_classes = 2, 
                               feature_extraction = True):

    # Load the pretrained faster r-cnn model.
    model = fasterrcnn_resnet50_fpn(pretrained = True)
    # If True, the pre-trained weights will be frozen.
    if feature_extraction == True:
        for p in model.parameters():
            p.requires_grad = False
    # Replace the original 91 class top layer with a new layer
    # tailored for num_classes.
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats,
                                                   num_classes)
    return model

def xml_to_dict(xml_path):
    # Decode the .xml file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Return the image size, object label and bounding box 
    # coordinates together with the filename as a dict.
    root.find
    return {"filename": xml_path,
            "image_width": int(root.find("./size/width").text),
            "image_height": int(root.find("./size/height").text),
            "image_channels": int(root.find("./size/depth").text),
            "labels": [label.text for label in root.findall("./object/name")],
            "x1s": [int(x1.text) for x1 in root.findall("./object/bndbox/xmin")],
            "y1s": [int(y1.text) for y1 in root.findall("./object/bndbox/ymin")],
            "x2s": [int(x2.text) for x2 in root.findall("./object/bndbox/xmax")],
            "y2s": [int(y2.text) for y2 in root.findall("./object/bndbox/ymax")]}


label_dict = {"lego": 1, "legod": 1}
reverse_label_dict = {1: "lego"}

class LegoDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=0, transforms = None):
        modes = ["train/", "val/", "test/"]
        self.root = root
        self.transforms = transforms
        self.train = modes[train]
        self.files = sorted(os.listdir(self.root + "images/" + self.train))
        for i in range(len(self.files)):
            self.files[i] = self.files[i].split(".")[0]
            self.label_dict = label_dict
    def __getitem__(self, i):
        # Load image from the hard disc.
        img = Image.open(os.path.join(self.root, 
              "images/" + self.train + self.files[i] + ".jpg")).convert("RGB")
        # Load annotation file from the hard disc.
        ann = xml_to_dict(os.path.join('../dataset/', self.files[i] + ".xml"))            
        # The target is given as a dict.
        target = {}
        target["boxes"] = []
        target["boxes"] = torch.as_tensor([], dtype=torch.float32).reshape(0, 4)
        target["labels"]=torch.as_tensor([], dtype = torch.int64)
        
        for i in range(len(ann["labels"])):
            target['boxes'] = torch.cat([target['boxes'], torch.as_tensor([float(ann["x1s"][i]), 
                                           float(ann["y1s"][i]), 
                                           float(ann["x2s"][i]), 
                                           float(ann["y2s"][i])], 
                                          dtype=torch.float32).unsqueeze(0)], 0)
            target['labels'] = torch.cat((target['labels'], torch.as_tensor(label_dict[ann["labels"][i]], dtype = torch.int64).unsqueeze(0)), 0)
            
        # Apply any transforms to the data if required.
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    def __len__(self):
        return len(self.files)

def get_transform():
    """
    Transforms a PIL Image into a torch tensor, and performs
    random horizontal flipping of the image if training a model.
    Inputs
        train: bool
            Flag indicating whether model training will occur.
    Returns
        compose: Compose
            Composition of image transforms.
    """
    transforms = []
    # ToTensor is applied to all images.
    transforms.append(ToTensor())

    return Compose(transforms)

def unbatch(batch, device):
    X, y = batch
    X = [x.to(device) for x in X]
    y = [{k: v.to(device) for k, v in t.items()} for t in y]
    return X, y
def train_batch(batch, model, optimizer, device):
    """
    Uses back propagation to train a model.
    Inputs
        batch: tuple
            Tuple containing a batch from the Dataloader.
        model: torch model
        optimizer: torch optimizer
        device: str
            Indicates which device (CPU/GPU) to use.
    Returns
        loss: float
            Sum of the batch losses.
        losses: dict
            Dictionary containing the individual losses.
    """
    model.train()
    X, y = unbatch(batch, device = device)
    optimizer.zero_grad()
    losses = model(X, y)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optimizer.step()
    return loss, losses
@torch.no_grad()
def validate_batch(batch, model, optimizer, device):
    model.train()
    X, y = unbatch(batch, device = device)
    optimizer.zero_grad()
    losses = model(X, y)
    loss = sum(loss for loss in losses.values())
    return loss, losses

def train_fasterrcnn(model, 
                 optimizer, 
                  n_epochs, 
              train_loader, 
        test_loader = None, 
                log = None, 
               keys = None, 
            device = "cpu"):
    """
    Trains a FasterRCNN model using train and validation 
    Dataloaders over n_epochs. 
    Returns a Report on the training and validation losses.
    Inputs
        model: FasterRCNN
        optimizer: torch optimizer
        n_epochs: int
            Number of epochs to train.
        train_loader: DataLoader
        test_loader: DataLoader
        log: Record
            torch_snippet Record to record training progress.
        keys: list
            List of strs containing the FasterRCNN loss names.
        device: str
            Indicates which device (CPU/GPU) to use.
    Returns
        log: Record
            torch_snippet Record containing the training records.
    """
    if log is None:
        log = Report(n_epochs)
    if keys is None:
        # FasterRCNN loss names.
        keys = ["loss_classifier", 
                   "loss_box_reg", 
                "loss_objectness", 
               "loss_rpn_box_reg"]
    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        N = len(train_loader)
        for ix, batch in enumerate(train_loader):
            loss, losses = train_batch(batch, model, 
                                  optimizer, device)
            # Record the current train loss.
            pos = epoch + (ix + 1) / N
            log.record(pos = pos, trn_loss = loss.item(), 
                       end = "\r")
        if test_loader is not None:
            N = len(test_loader)
            for ix, batch in enumerate(test_loader):
                loss, losses = validate_batch(batch, model, 
                                         optimizer, device)
                if loss < best_val_loss:
                    best_val_loss = loss
                    save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                    torch.save(save_dict, "best.pth")

	     	# Save latest model
                save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(save_dict, "last.pth")
                # Record the current validation loss.
                pos = epoch + (ix + 1) / N
                log.record(pos = pos, val_loss = loss.item(), 
                           end = "\r")
    log.report_avgs(epoch + 1)
    return log

@torch.no_grad()
def predict_batch(batch, model, device):
    model.to(device)
    model.eval()
    X, y = unbatch(batch, device = device)
    predictions = model(X)
    return predictions, y
def predict(model, data_loader, device = "cpu"):
    images = []
    predictions = []
    for i, batch in enumerate(data_loader):
        if i == 50:
            break
        X, p = predict_batch(batch, model, device)
        images = images + X
        predictions = predictions + p
    
    return images, predictions


def decode_prediction(prediction, 
                      score_threshold = 0.8, 
                      nms_iou_threshold = 0.2):
    """
    Inputs
        prediction: dict
        score_threshold: float
        nms_iou_threshold: float
    Returns
        prediction: tuple
    """
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]
    # Remove any low-score predictions.
    if score_threshold is not None:
        want = scores > score_threshold
        boxes = boxes[want]
        scores = scores[want]
        labels = labels[want]
    # Remove any overlapping bounding boxes using NMS.
    if nms_iou_threshold is not None:
        want = torchvision.ops.nms(boxes = boxes, scores = scores, 
                                iou_threshold = nms_iou_threshold)
        boxes = boxes[want]
        scores = scores[want]
        labels = labels[want]
    return (boxes.cpu().numpy(), 
            labels.cpu().numpy(), 
            scores.cpu().numpy())

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

training_targets = []
validation_targets = []
names = [x.split('.')[0] for x in os.listdir(images_path + 'train/')]
training_images = [images_path + 'train/' + x for x in os.listdir(images_path + 'train/')]
validation_images = [images_path + 'val/' + x for x in os.listdir(images_path + 'val/')]
test_images = [images_path + 'test/' + x for x in os.listdir(images_path + 'test/')]


train_ds = LegoDataset(root_path, train = 0, transforms=get_transform())

val_ds = LegoDataset(root_path, train = 1, transforms=get_transform())

test_ds = LegoDataset(root_path, train = 2, transforms=get_transform())

# Collate image-target pairs into a tuple.
def collate_fn(batch):
    return tuple(zip(*batch))
# Create the DataLoaders from the Datasets. 
train_dl = torch.utils.data.DataLoader(train_ds, 
                                 batch_size = 2, 
                                 shuffle = True, 
                        collate_fn = collate_fn)
val_dl = torch.utils.data.DataLoader(val_ds, 
                             batch_size = 2, 
                            shuffle = False, 
                    collate_fn = collate_fn)

test_dl = torch.utils.data.DataLoader(test_ds, 
                              batch_size = 1, 
                             shuffle = False, 
                     collate_fn = collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_object_detection_model(num_classes = 2,   
                        feature_extraction = False)
# Use the stochastic gradient descent optimizer.
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, 
                        lr = 0.005, 
                    momentum = 0.9, 
             weight_decay = 0.0005)

'''log = train_fasterrcnn(model = model, 
               optimizer = optimizer, 
                        n_epochs = 20,
             train_loader = train_dl, 
                test_loader = val_dl,
             log = None, keys = None,
                     device = device)
                     '''



def IOU(box1, box2):
	""" We assume that the box follows the format:
		box1 = [x1,y1,x2,y2], and box2 = [x3,y3,x4,y4],
		where (x1,y1) and (x3,y3) represent the top left coordinate,
		and (x2,y2) and (x4,y4) represent the bottom right coordinate """
	x1, y1, x2, y2 = box1	
	x3, y3, x4, y4 = box2
	x_inter1 = max(x1, x3)
	y_inter1 = max(y1, y3)
	x_inter2 = min(x2, x4)
	y_inter2 = min(y2, y4)
	width_inter = abs(x_inter2 - x_inter1)
	height_inter = abs(y_inter2 - y_inter1)
	area_inter = width_inter * height_inter
	width_box1 = abs(x2 - x1)
	height_box1 = abs(y2 - y1)
	width_box2 = abs(x4 - x3)
	height_box2 = abs(y4 - y3)
	area_box1 = width_box1 * height_box1
	area_box2 = width_box2 * height_box2
	area_union = area_box1 + area_box2 - area_inter
	iou = area_inter / area_union
	return iou

#load model
checkpoint = torch.load('best.pth')
model.load_state_dict(checkpoint['model'])

accs = []

for ix, batch in enumerate(test_dl):
    pred, true = predict_batch(batch, model, device)
    boxes_pred, _, _ = decode_prediction(pred[0])
    boxes_true = true[0]['boxes'].cpu().numpy()
    n = len(boxes_true)
    match = 0
    for box in boxes_pred:
        for true_box in boxes_true:
            iou = IOU(box, true_box)
            if iou > 0.8:
                match += 1
                boxes_true = np.delete(boxes_true, np.where(boxes_true == true_box)[0], axis=0)
                break
    accs.append(match / n)

    
    
map = sum(accs) / len(accs)

print(map)