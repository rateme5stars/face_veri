import torch.utils.data

from .trainers import Trainer
from face_verification.losses import Loss
from face_verification.metrics import Metric
from face_verification.models.face_verification import ArcFace
from face_verification.monitors import Monitor


class ArcFaceTrainer(Trainer):

    def __init__(
        self,
        arcface: ArcFace,
        embedder: torch.nn.Module,
        loss: Loss | None,
        optimizer: torch.optim.Optimizer | None = None,
        metrics: list[Metric] | None = None,
        monitors: list[Monitor] | None = None,
        device: torch.device | str = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self.arcface = arcface.to(device)
        super().__init__(embedder, loss, optimizer, metrics, monitors, device)

    def train_one_epoch(
        self, train_loader: torch.utils.data.DataLoader
    ) -> None:
        progress_bar = self.tqdm(train_loader)
        self.model.train()

        for input_batch, target_batch in progress_bar:
            self.train_one_step(input_batch, target_batch)
            progress_bar.set_description(
                f"Train: {self.format_results(train=True)}"
            )

        self.notify_monitors("train")
        self.reset()

    def train_one_step(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        if self.loss is None or self.optimizer is None:
            raise AttributeError(
                "Model can't be trained because loss or optimizer is None."
            )

        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)

        embedding_batch = self.model(input_batch)
        output_batch = self.arcface(embedding_batch, target_batch)
        computed_loss = self.loss(output_batch, target_batch)

        computed_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.loss.update(output_batch, target_batch)
        self.update(embedding_batch, target_batch)

    def update(
        self,
        output_batch: torch.Tensor,
        target_batch: torch.Tensor,
    ) -> None:
        for metric in self.metrics:
            metric.update(output_batch, target_batch)

    def format_results(self, train=False) -> str:
        loss_result_format = (
            "" if not train else f"loss = {self.loss.result():.4f}"
        )
        metric_results_format = "".join(
            f", {metric.name} = {metric.result():.4f}" for metric in self.metrics
        )
        results_format = loss_result_format + metric_results_format
        return results_format
