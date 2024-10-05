from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Generator
from torch.utils.data import Dataset
from torch.fft import fftn
from torch import Tensor
import os.path as osp
import pandas as pd
import torch
import os

metadata: Dict[str, Dict[str,str]] = dict(
    lunar = dict(
        catalog = 'data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv',
        train_path = 'data/mars/training/data',
        test_path = 'data/mars/test/data',
    ),
    mars = dict(
        catalog = 'data/mars/training/catalogs/Mars_InSight_training_catalog_final.csv',
        train_path = 'data/lunar/training/data/S12_GradeA',
        test_path = 'data/lunar/test/data'
    )
)

def recursive_search(parent: str) -> Generator[str, None, None]:
    for child in os.listdir(parent):
        child_path = os.path.join(parent, child)
        if os.path.isdir(child_path):
            yield from recursive_search(child_path)
        elif child.endswith('.csv'):
            yield child_path

class TestDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.filepaths = [filename for filename in recursive_search('./data/mars/')] + [filename for filename in recursive_search('./data/lunar/')]
        self.dataset = torch.from_numpy(pd.concat(
            [
                pd.DataFrame(filepath) for filepath in self.filepaths
            ], axis = 0
        ).values)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tensor:
        return self.dataset[idx]

class TrainDataset(Dataset):
    def __init__(self) -> None:
        self.meta_lunar: pd.DataFrame = pd.read_csv(metadata['lunar']['catalog']),
        self.meta_mars: pd.DataFrame = pd.read_csv(metadata['mars']['catalog'])
        self.metadata: List[str] = list(pd.concat(
            [
                self.meta_lunar['filename'],
                self.meta_mars['filename']
            ],
            axis = 0
        ).values)

        self.data: Dict[str, Tensor] = {
            filename: self.preprocess_csv(filename, 'lunar') for filename in self.meta_lunar['filename'].values
        }.update(
            {
                filename: self.preprocess_csv(f'{filename}.csv', 'mars') for filename in self.meta_mars['filename'].values
            }
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def preprocess_csv(self, name: str, type_: str) -> Tensor:
        path: str = osp.join(metadata[type_]['catalog'], name)
        df: pd.DataFrame = pd.read_csv(path)
        ## add more preprocessing steps
        out: Tensor = torch.from_numpy(df.values)
        ### add the wavelet / fourier transform is needed
        return out

    def __getitem__(self, idx: int) -> Tensor:
        filename: str = self.metadata[idx]
        return self.data[filename]

class DataModule(LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int, pin_memory: bool, train_pt: float) -> None:
        super().__init__()
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory
        self.train_pt: float = train_pt

    def setup(self, stage: str) -> None:
        self.train_ds = TrainDataset()
        self.test_ds = TestDataset()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers)

