from typing import Optional

import pytorch_lightning as pl
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from src.config import get_cfg_defaults
from src.data.datasets.build_dataset import build_dataset
from src.data.utils import *


class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.TRAINER.BATCH_SIZE

        self.dataset_train_val = None
        self.train_dataset = None
        self.val_dataset = None
        self.predict_dataset = None

    def prepare_data(self):
        self.dataset_train_val = build_dataset(self.cfg, is_train=True)
        self.dataset_train_val.prepare_data()
        self.predict_dataset = build_dataset(self.cfg, is_train=False)
        self.predict_dataset.prepare_data()

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            kf = KFold(n_splits=self.cfg.DATAMODULE.NUM_CV_SPLITS)
            train, val = list(kf.split(np.arange(len(self.dataset_train_val))))[self.cfg.DATAMODULE.I_CV_FOLD]
            self.train_dataset = Subset(self.dataset_train_val, train)
            self.val_dataset = Subset(self.dataset_train_val, val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)


    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        ...

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    # cfg.merge_from_list(['DATASET.TRANSFORM', 'precomputed_flow'])
    dm = MyDataModule(cfg)
    dm.prepare_data()
    dm.setup()
    vid, fmri = dm.train_dataset.__getitem__(0)
    vid2 = dm.predict_dataset.__getitem__(0)
    ...
