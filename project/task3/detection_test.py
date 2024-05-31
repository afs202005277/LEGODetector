from ultralytics import YOLO
import os
folder_name = "results"


model = YOLO('best.pt')

imgs = '../generated_dataset/'
otherimgs = 'datasets/vc/images/val/'

metrics = model.val(data='datasets/vc/data.yaml')