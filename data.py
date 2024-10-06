from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Generator, Tuple
from torch.utils.data import Dataset
from torch import Tensor
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
        self.filepaths = [filename for filename in recursive_search('./data/mars/test')] + \
                        [filename for filename in recursive_search('./data/lunar/test')]
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
    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length
        self.filepaths = [filename for filename in recursive_search('./data/mars/training/data')] + \
                        [filename for filename in recursive_search('./data/lunar/training/data')]
        self.meta_lunar: pd.DataFrame = pd.read_csv(metadata['lunar']['catalog'], index_col = ['filename']),
        self.meta_mars: pd.DataFrame = pd.read_csv(metadata['mars']['catalog'], index_col = ['filename'])
        self.metadata: pd.DataFrame = pd.concat(
            [
                self.meta_lunar,
                self.meta_mars,
            ], axis = 0
        )
        self.padding = lambda input: ## padding(input, sequence_length)

    def preprocessing(self) -> None:
        self.dataset: List[Tuple[Tensor, Tensor]] = []
        for file in self.filepaths:
            try:
                arrive = self.metadata.loc[['time_rel(sec)'], file]
                out: Tensor = self.get_data(file)
                self.dataset.extend((input, target)) ### mirar
            except IndexError:
                continue

    def __len__(self) -> int:
        return len(self.metadata)

    def get_data(self, file: str) -> List[Tuple[Tensor, Tensor]]: ## atencion
        ## get target (tensor de 0 y 1s tal que el idx del arrival este coincidiendo con el arrival real)
        ## arrival_time = self.metadata[file].iloc['time_rel(sec)']
        ## creas el tensor

        df: pd.DataFrame = pd.read_csv(file, parse_dates =['time_abs(%Y-%m-%dT%H:%M:%S.%f)'] ,index_col = ['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])
        velocity: Tensor = torch.from_numpy(torch.from_numpy(df.resample('s').mean().values))
        max: Tensor = torch.from_numpy(df.resample('s').max().values)
        min: Tensor = torch.from_numpy(df.resample('s').min().values)
        ### add the wavelet / fourier transform is needed

        ### separar de a sequence_length
        ### input(sequence_length, input_size) -> target(sequence_length) (0, 0, 1, 0)
        return self.padding(torch.stack(
            [
                velocity, accelaration, butter, sfft ## convolution transpuest
            ], dim = -1
        ))

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        filename: str = self.metadata[idx]
        return self.data[filename]

class DataModule(LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int, pin_memory: bool) -> None:
        super().__init__()
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory

    def setup(self, stage: str) -> None:
        self.train_ds = TrainDataset()
        self.test_ds = TestDataset()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers)

