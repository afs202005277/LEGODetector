import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random
# Load a pretrained model


folder_name = "results"

base_folder = "../"
image_paths_file = base_folder + "dataset/"
model = YOLO('YOLOv8n.pt')
'''
image_names = [x.split('.')[0] for x in os.listdir(image_paths_file) if not x.endswith('.xml')]

print(image_names)

#randomize images order

random.shuffle(image_names)

# divide images in training and validation set (60% training, 40% validation)
split = int(0.6 * len(image_names))
training_images = image_names[:split]
validation_images = image_names[split:]

os.makedirs('train', exist_ok=True)
os.makedirs('val', exist_ok=True)
os.makedirs('train_label', exist_ok=True)
os.makedirs('val_label', exist_ok=True)

for image in training_images:
    os.rename(image_paths_file + image + '.jpg', 'train/' + image + '.jpg')
    os.rename(image_paths_file + image + '.xml', 'train_label/' + image + '.xml')

for image in validation_images:
    os.rename(image_paths_file + image + '.jpg', 'val/' + image + '.jpg')
    os.rename(image_paths_file + image + '.xml', 'val_label/' + image + '.xml')'''

#train yolo model
model.train(data='datasets/vc/data.yaml', epochs=10, batch=16)
    
    
