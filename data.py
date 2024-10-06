from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Generator, Tuple
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset
from datetime import timedelta
from torch import Tensor
import pandas as pd
import torch
import os

metadata: Dict[str, Dict[str,str]] = dict(
    lunar = dict(
        catalog = '../data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv',
        train_path = '../data/mars/training/data',
        test_path = '../data/mars/test/data',
    ),
    mars = dict(
        catalog = '../data/mars/training/catalogs/Mars_InSight_training_catalog_final.csv',
        train_path = '../data/lunar/training/data/S12_GradeA',
        test_path = '../data/lunar/test/data'
    )
)

def recursive_search(parent: str) -> Generator[str, None, None]:
    for child in os.listdir(parent):
        child_path = os.path.join(parent, child)
        if os.path.isdir(child_path):
            yield from recursive_search(child_path)
        elif child.endswith('.csv'):
            yield child_path

def mars_prep(filepaths: List[str]) -> List[pd.DataFrame]:
    out = []
    for filepath in filepaths:
        df = pd.read_csv(filepath, parse_dates = ['time(%Y-%m-%dT%H:%M:%S.%f)'], index_col =['time(%Y-%m-%dT%H:%M:%S.%f)'])
        df['norm_v'] = StandardScaler().fit_transform(df['velocity(c/s)'].values.reshape(-1, 1))
        df.columns = ['time_abs(%Y-%m-%dT%H:%M:%S.%f)','time_rel(sec)','velocity(m/s)', 'norm_v']
        out.append(df)
    return out

def lunar_prep(filepaths: List[str]) -> List[pd.DataFrame]:
    out = []
    for filepath in filepaths:
        df = pd.read_csv(filepath, parse_dates = ['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], index_col =['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])
        df['norm_v'] = StandardScaler().fit_transform(df['velocity(m/s)'].values.reshape(-1, 1))
        out.append(df)
    return out

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


class Base(Dataset):
    def __init__(self, sequence_length: int, resample: timedelta, train: bool) -> None:
        self.sequence_length = sequence_length
        self.resample = pd.Timedelta(resample)
        if train:
            self.filepaths = [filename for filename in recursive_search('../data/lunar/training/data')] + \
                            [filename for filename in recursive_search('../data/mars/training/data')]
        else:
            self.filepaths = [filename for filename in recursive_search('../data/lunar/test/data')] + \
                            [filename for filename in recursive_search('../data/mars/test/data')]

        self.meta_lunar: pd.DataFrame = pd.read_csv(metadata['lunar']['catalog'], index_col = ['filename'])
        self.meta_mars: pd.DataFrame = pd.read_csv(metadata['mars']['catalog'], index_col = ['filename'])
        self.metadata: pd.DataFrame = pd.concat(
            [
                self.meta_lunar,
                self.meta_mars,
            ], axis = 0
        )
        self.preprocessing()

    def zero_padding(self, x: Tensor) -> Tensor:
        pad_size = self.sequence_length - x.size(-1)
        if pad_size > 0:
            return torch.nn.functional.pad(x, (0, pad_size))
        return x

    def preprocessing(self) -> None:
        self.data: List[Tuple[Tensor, Tensor]] = []
        for file in self.filepaths:
            try:
                out: Tensor = self.get_data(file)
                self.data.extend(out)
            except IndexError:
                continue

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.data[idx]

class TrainDataset(Base):
    def __init__(self, sequence_length: int, resample: timedelta) -> None:
        super().__init__(sequence_length, resample, True)

    def get_data(self, file: str) -> List[Tuple[Tensor, Tensor]]: ## atencion
        df: pd.DataFrame = pd.read_csv(file, parse_dates =['time_abs(%Y-%m-%dT%H:%M:%S.%f)'] ,index_col = ['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])
        target_idx: int = round(timedelta(seconds=self.metadata["time_rel(sec)"].loc[os.path.basename(file)])/self.resample)
        ## vel
        velocities: Tensor = torch.from_numpy(df["norm_v"].resample(self.resample).mean().values)
        times: Tensor = torch.from_numpy(df["time_rel(sec)"].resample(self.resample).mean().values)

        ## target
        target = torch.zeros(velocities.shape[-1])
        target[target_idx] = 1

        ## vel, acceleration, butterworth, wavelet, fourier
        acceleration: Tensor = (velocities[:-1] - velocities[1:]) / (times[:-1] - times[1:])
        max_velocities: Tensor = torch.from_numpy(df["norm_v"].resample(self.resample).max().values)
        butterworth_bandpass: Tensor = torch.from_numpy(butter_bandpass(max_velocities, 0.1, 0.5, fs = 60))
        wavelet_transform: Tensor = ...
        sliding_fourier: Tensor = ...

        out: Tensor = torch.stack(
            [
                velocities,
                acceleration,
                max_velocities,
                butterworth_bandpass,
                wavelet_transform,
                sliding_fourier,
            ], dim = -1
        )

        return [
            (self.zero_padding(input), self.zero_padding(output)) \
            for input, output in zip(out.split(self.sequence_length), target.split(self.sequence_length))
        ]

class TestDataset(Base):
    def __init__(self, sequence_length: int, resample: timedelta) -> None:
        super().__init__(sequence_length, resample, False)
    def get_data(self, file: str) -> List[Tensor]: ## atencion
        df: pd.DataFrame = pd.read_csv(file, parse_dates =['time_abs(%Y-%m-%dT%H:%M:%S.%f)'] ,index_col = ['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])
        ## vel
        velocities: Tensor = torch.from_numpy(df["norm_v"].resample(self.resample).mean().values)
        times: Tensor = torch.from_numpy(df["time_rel(sec)"].resample(self.resample).mean().values)

        ## vel, acceleration, butterworth, wavelet, fourier
        acceleration: Tensor = (velocities[:-1] - velocities[1:]) / (times[:-1] - times[1:])
        max_velocities: Tensor = torch.from_numpy(df["norm_v"].resample(self.resample).max().values)
        butterworth_bandpass: Tensor = torch.from_numpy(butter_bandpass(max_velocities, 0.1, 0.5, fs = 60))
        wavelet_transform: Tensor = ... ### default
        sliding_fourier: Tensor = ... ### trans conv

        out: Tensor = torch.stack(
            [
                velocities,
                acceleration,
                max_velocities,
                butterworth_bandpass,
                wavelet_transform,
                sliding_fourier,
            ], dim = -1
        )

        return [
            self.zero_padding(input) for input in zip(out.split(self.sequence_length))
        ]



class DataModule(LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int, pin_memory: bool) -> None:
        super().__init__()
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory

    def setup(self, stage: str) -> None:
        self.train_ds = TrainDataset(30, timedelta(minutes = 1))
        self.test_ds = TestDataset(30, timedelta(minutes = 1))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers)
