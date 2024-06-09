import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.data.annotator import auto_annotate
import random



'''auto_annotate(data='datasets/vc/images/test', det_model='best_small.pt')
auto_annotate(data='datasets/vc/images/train', det_model='best_small.pt')
auto_annotate(data='datasets/vc/images/val', det_model='best_small.pt')'''


model1 = YOLO('yolov8n-seg.pt', task='segmentation')
model1.train(data='datasets/seg/data.yaml', epochs=50, batch=-1, model="best_nano.pt", dropout=0.1)

model2 = YOLO('yolov8s-seg.pt', task='segmentation')
model2.train(data='/datasets/seg/data.yaml', epochs=50, batch=-1, model="best_small.pt", dropout=0.1)

model3 = YOLO('yolov8m-seg.pt', task='segmentation')
model3.train(data='datasets/seg/data.yaml', epochs=50, batch=-1, model="best_medium.pt", dropout=0.1)

model4 = YOLO('yolov8l-seg.pt', task='segmentation')
model4.train(data='datasets/seg/data.yaml', epochs=50, batch=-1, model="best_large.pt", dropout=0.1)
