import torch.utils.data

from .trainers import Trainer
from face_verification.losses import Loss, CrossEntropy
from face_verification.metrics import Metric
from face_verification.monitors import Monitor


class ArcFaceTrainer(Trainer):

    def __init__(
        self,
        arcface: torch.nn.Module,
        loss: Loss | None = CrossEntropy(),
        optimizer: torch.optim.Optimizer | None = None,
        metrics: list[Metric] | None = None,
        monitors: list[Monitor] | None = None,
        device: torch.device | str = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        super().__init__(arcface, loss, optimizer, metrics, monitors, device)

    def train_one_step(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        if self.loss is None or self.optimizer is None:
            raise AttributeError(
                "Model can't be trained because loss or optimizer is None."
            )

        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)

        output_batch = self.model(input_batch, target_batch)
        computed_loss = self.loss(output_batch, target_batch)

        computed_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.update(output_batch, target_batch)

    def valid_one_epoch(
        self, valid_loader: torch.utils.data.DataLoader
    ) -> None:
        pass

    def test(self, test_loader: torch.utils.data.DataLoader) -> None:
        pass
