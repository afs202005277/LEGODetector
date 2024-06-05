import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.data.annotator import auto_annotate
import random

model = YOLO('yolov8l-seg.pt', task='segmentation')


'''auto_annotate(data='datasets/vc/images/test', det_model='best_small.pt')
auto_annotate(data='datasets/vc/images/train', det_model='best_small.pt')
auto_annotate(data='datasets/vc/images/val', det_model='best_small.pt')'''


model.train(data='datasets/seg/data.yaml', epochs=100, batch=-1, model="best_large_check.pt", dropout=0.1)
