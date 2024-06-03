import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random
# Load a pretrained model


folder_name = "results"

base_folder = "../"
image_paths_file = base_folder + "dataset/"
model = YOLO('yolov8l.pt')

#train yolo model
model.train(data='datasets/vc/data.yaml', epochs=50, batch=-1, model="best_medium_check.pt", dropout=0.1)
    
    
