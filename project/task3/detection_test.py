from ultralytics import YOLO
import os
folder_name = "results"

imgs = '../generated_dataset/'
otherimgs = 'datasets/vc/images/val/'

model = YOLO('best_nano.pt')
metrics = model.val(data='datasets/vc/data.yaml', split="test", batch=1)

model = YOLO('best_small.pt')
metrics = model.val(data='datasets/vc/data.yaml', split="test", batch=1)

model = YOLO('best_medium.pt')
metrics = model.val(data='datasets/vc/data.yaml', split="test", batch=1)

model = YOLO('best_large.pt')
metrics = model.val(data='datasets/vc/data.yaml', split="test", batch=1)
