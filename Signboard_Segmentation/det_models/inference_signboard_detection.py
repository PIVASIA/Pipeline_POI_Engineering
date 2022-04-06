import pytorch_lightning as pl
import numpy as np
import os
from PIL import Image,ImageDraw
from datasets_signboard_detection.dataset import Labelizer

class Color_convert():
    def __init__(self):
        super().__init__()
        self.labels = {'bien': "red"}
        
    def transform(self, label):
        return self.labels[label]
    
    def num_classes(self):
        return len(self.labels)

def compose(output, mask):
    w,h = mask.shape

    for i in range(0, w):
        for j in range(0,h):
            if (mask[i,j] > 0.5):
                output[i,j] = 1
    return output

class POIDetectionTask(pl.LightningModule):
    def __init__(self,
                 model,
                 data_path):
        super().__init__()
        
        self.model = model
        self.data_path = data_path
        self.output = []
        self.pred_targets = []
        self.pred_masks = []
        self.pred_labels = []
        self.labelizer = Labelizer()
        self.color_convert = Color_convert()
        self.num_sticker = 0

    def forward(self, x):
        output = self.model(x)
        return output

    def predict_step(self, test_batch, batch_idx):
        images, targets = test_batch
        outputs = self(images)
        for target in outputs:
            shape = target['boxes']
            masks = target['masks']
            score = target['scores']
            labels = target['labels']
            shape = shape.numpy()
            masks = masks.numpy()
            score = score.numpy()
            labels = labels.numpy()
            select_shape = []
            select_masks = []
            select_labels = []
            for i in range(0,len(score)):
                if (score[i]>0.7):
                    select_shape.append(shape[i])
                    select_masks.append(masks[i])
                    select_labels.append(labels[i])
            select_shape = [select_shape]
            select_masks = [select_masks]
            select_labels = [select_labels]
            self.pred_targets = select_shape
            self.pred_masks = select_masks
            self.pred_labels = select_labels
    
    def on_predict_end(self):
        image = Image.open(self.data_path).convert('RGBA')

        pred_boxes = self.pred_targets[0]
        pred_masks = self.pred_masks[0]
        pred_labels = self.pred_labels[0]
    
        width, height = image.size
        self.output = np.zeros((height,width), dtype="uint8")
        for j in range(0,len(pred_masks)):
            mask = pred_masks[j]
            self.output = compose(self.output, mask[0])
        