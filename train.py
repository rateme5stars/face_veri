from pathlib import Path

import torch

from face_verification.trainers import ArcFaceTrainer
from face_verification.preprocessing import (
    split_identities,
    find_unique_identities,
    find_paths_with_identities,
)
from face_verification.metrics import Accuracy
from face_verification.dataloaders import create_dataloader
from face_verification.models.face_verification import ArcFace, MobileNetV2


IMAGE_DIR = Path("meglass/")
VALID_SIZE = 0.1
TEST_SIZE = 0.1
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
RANDOM_STATE = 12
N_EPOCHS = 1
LEARNING_RATE = 1e-5


def main() -> None:
    image_paths = list(IMAGE_DIR.glob("*.jpg"))
    unique_identities = find_unique_identities(image_paths)

    train_identities, valid_identities, test_identities = split_identities(
        unique_identities, VALID_SIZE, TEST_SIZE, RANDOM_STATE
    )

    train_paths = find_paths_with_identities(image_paths, train_identities)
    valid_paths = find_paths_with_identities(image_paths, valid_identities)
    test_paths = find_paths_with_identities(image_paths, test_identities)

    train_loader = create_dataloader(train_paths, IMAGE_SIZE, BATCH_SIZE)
    valid_loader = create_dataloader(valid_paths, IMAGE_SIZE, BATCH_SIZE)
    test_loader = create_dataloader(test_paths, IMAGE_SIZE, BATCH_SIZE)

    backbone = MobileNetV2()
    arcface = ArcFace(backbone, len(train_identities))
    trainable_params = [
        param for param in arcface.parameters() if param.requires_grad
    ]

    optimizer = torch.optim.Adam(trainable_params, LEARNING_RATE)

    trainer = ArcFaceTrainer(
        arcface,
        optimizer=optimizer,
        metrics=[Accuracy()]
    )
    trainer.train(train_loader, valid_loader, N_EPOCHS)
    trainer.test(test_loader)


if __name__ == "__main__":
    main()
