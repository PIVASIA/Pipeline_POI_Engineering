import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets_signboard_detection.dataset import PoIDataset
import datasets_signboard_detection.utils as utils



def _get_name(filepath):
    images = filepath
    return images

class POIDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_path: str,
                 train_batch_size=8,
                 test_batch_size=8,
                 seed=28):
        super().__init__()
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.seed = seed
    

    def prepare_data(self):
        pass
    
    def setup(self, stage="fit"):
        transform = [transforms.ToTensor()]
        test_transform = transforms.Compose(transform)
        if stage == "predict" or stage is None:
            self.test_dataset = PoIDataset(self.data_path, 
                                           transforms=test_transform)

    def predict_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset,
                              batch_size=self.test_batch_size,
                              shuffle=False,
                              num_workers=2,
                              collate_fn=utils.collate_fn)