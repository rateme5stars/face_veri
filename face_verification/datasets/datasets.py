from pathlib import Path
from typing import Callable

import numpy as np
import torch.utils.data
import torchvision
from sklearn.preprocessing import OneHotEncoder

from face_verification.preprocessing import identity_from_name


class MeGlass(torch.utils.data.Dataset):

    def __init__(
        self,
        image_paths: list[Path],
        image_size: tuple[int, int],
        train: bool,
    ) -> None:
        self._image_paths = image_paths
        self._transform = create_transform(image_size, train)
        identities = [
            identity_from_name(path.name) for path in image_paths
        ]
        self._labels = one_hot_encode(identities)

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self._image_paths[idx]
        image = self._transform(torchvision.io.read_image(str(image_path)))
        label = self._labels[idx]
        return image, label


def one_hot_encode(identities: list[str]) -> torch.Tensor:
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    return torch.from_numpy(
        one_hot_encoder.fit_transform(np.array(identities)[:, np.newaxis])
    )


def create_transform(
    image_size: tuple[int, int], train: bool
) -> Callable[[torch.Tensor], torch.Tensor]:
    transforms = [
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.Lambda(lambda image: image / 255),
    ]
    if train:
        transforms.extend(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomGrayscale(),
            ]
        )
    return torchvision.transforms.Compose(transforms)
