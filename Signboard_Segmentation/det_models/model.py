import torch.optim as optim
import pytorch_lightning as pl
from det_models.backbone import initialize_model

class POIDetection(pl.LightningModule):
    def __init__(self,
                 n_classes,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model, _ = initialize_model(kwargs["backbone"], 
                                         n_classes, 
                                         tune_only=kwargs["tune_fc_only"])
        
    def forward(self, images, targets=None):
        images = list(image for image in images)
        if targets is not None :
            targets = [{k: v for k, v in t.items()} for t in targets]
            outputs = self.model(images, targets)
        else:
            outputs = self.model(images)
        return outputs
