import numpy as np
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from det_models.model import POIDetection
from datasets_signboard_detection.datamodule import POIDataModule
import cv2

def load_model(checkpoint_path):
    model = POIDetection.load_from_checkpoint(checkpoint_path=checkpoint_path)
    return model

def inference_signboard(image_path):

    dm = POIDataModule(data_path=image_path,
                       seed=42)
    dm.setup("predict")

    model = load_model("checkpoints/checkpoint29.ckpt")
    from det_models.inference_signboard_detection import POIDetectionTask
    task = POIDetectionTask(model,
                            data_path=image_path)

    trainer = pl.Trainer(gpus=0)
    trainer.predict(task, datamodule=dm)
    return task.output