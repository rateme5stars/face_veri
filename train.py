import itertools
from pathlib import Path

import torch

from face_verification.preprocessing import (
    split_identities,
    find_unique_identities,
    find_paths_with_identities,
)
from face_verification.dataloaders import create_dataloader
from face_verification.losses import CrossEntropy
from face_verification.metrics import EmbeddingAccuracy
from face_verification.monitors import EarlyStopping, ModelCheckpoint
from face_verification.models.face_verification import ArcFace, MobileNetV2
from face_verification.trainers import ArcFaceTrainer


IMAGE_DIR = Path("meglass/")
CHECKPOINT_DIR = Path("checkpoint/")
VALID_SIZE = 0.1
TEST_SIZE = 0.1
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
RANDOM_STATE = 12
N_EPOCHS = 100
LEARNING_RATE = 1e-5


def main() -> None:
    image_paths = list(IMAGE_DIR.glob("*.jpg"))
    unique_identities = find_unique_identities(image_paths)

    train_identities, test_identities, valid_identities = split_identities(
        unique_identities, VALID_SIZE, TEST_SIZE, RANDOM_STATE
    )

    train_paths = find_paths_with_identities(image_paths, train_identities)
    test_paths = find_paths_with_identities(image_paths, test_identities)
    valid_paths = find_paths_with_identities(image_paths, valid_identities)

    train_loader = create_dataloader(
        train_paths, IMAGE_SIZE, BATCH_SIZE, train=True
    )
    test_loader = create_dataloader(
        test_paths, IMAGE_SIZE, BATCH_SIZE, train=False
    )
    valid_loader = create_dataloader(
        valid_paths, IMAGE_SIZE, BATCH_SIZE, train=False
    )

    arcface = ArcFace(len(train_identities))
    embedder = MobileNetV2()
    trainable_params = [
        param for param in
        itertools.chain(arcface.parameters(), embedder.parameters())
        if param.requires_grad
    ]

    loss = CrossEntropy()
    optimizer = torch.optim.Adam(trainable_params, LEARNING_RATE)
    metrics = [EmbeddingAccuracy()]
    monitors = [
        ModelCheckpoint(loss, CHECKPOINT_DIR),
        EarlyStopping(loss, patience=10),
    ]

    trainer = ArcFaceTrainer(
        arcface, embedder, loss, optimizer, metrics, monitors
    )
    trainer.train(train_loader, valid_loader, N_EPOCHS)
    trainer.test(test_loader)


if __name__ == "__main__":
    main()
