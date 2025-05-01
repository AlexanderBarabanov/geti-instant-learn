from dataclasses import dataclass

import numpy as np
from efficientvit.diffusioncore.data_provider.base import (
    BaseDataProvider,
    BaseDataProviderConfig,
)
from torch.utils.data import Dataset
from torchvision.datasets.folder import DatasetFolder

__all__ = ["LatentImageNetDataProviderConfig", "LatentImageNetDataProvider"]


@dataclass
class LatentImageNetDataProviderConfig(BaseDataProviderConfig):
    name: str = "latent_imagenet"
    data_dir: str = "assets/data/latent/dc_ae_f32c32/imagenet_512"


class LatentImageNetDataProvider(BaseDataProvider):
    def __init__(self, cfg: LatentImageNetDataProviderConfig):
        super().__init__(cfg)
        self.cfg: LatentImageNetDataProviderConfig

    def build_datasets(self) -> tuple[Dataset, Dataset | None, Dataset | None]:
        train_dataset = DatasetFolder(self.cfg.data_dir, np.load, [".npy"])
        return train_dataset, None, None
