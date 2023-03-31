from pathlib import Path

import torch.utils.data

from face_verification.datasets import MeGlass


def create_dataloader(
    image_paths: "list[Path]",
    image_size: "tuple[int, int]",
    batch_size: int,
) -> torch.utils.data.DataLoader:
    dataset = MeGlass(image_paths, image_size, train=True)
    return torch.utils.data.DataLoader(dataset, batch_size)
