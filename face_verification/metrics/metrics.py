from abc import ABC, abstractmethod
import torch

from face_verification import one_hot_decode


class Metric(ABC):

    @abstractmethod
    def __call__(
        self, target_batch: torch.Tensor, output_batch: torch.Tensor
    ) -> float:
        pass

    @abstractmethod
    def update(
        self, target_batch: torch.Tensor, output_batch: torch.Tensor
    ) -> None:
        pass

    @abstractmethod
    def result(self) -> float:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def merge(self, other) -> None:
        pass


class Accuracy(Metric):

    def __init__(self) -> None:
        self.name = "accuracy"
        self.prediction_count = 0
        self.correct_prediction_count = 0

    def __call__(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> float:
        n_predictions, *_ = output_batch.shape
        n_correct_predictions = (
            one_hot_decode(output_batch) == one_hot_decode(target_batch)
        ).sum().item()
        return n_correct_predictions / n_predictions

    def update(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        batch_size, *_ = output_batch.shape
        self.prediction_count += batch_size
        self.correct_prediction_count += (
            one_hot_decode(output_batch) == one_hot_decode(target_batch)
        ).sum().item()

    def result(self) -> float:
        return self.correct_prediction_count / self.prediction_count

    def reset(self) -> None:
        self.prediction_count = 0
        self.correct_prediction_count = 0

    def merge(self, other) -> None:
        self.prediction_count += other.prediction_count
        self.correct_prediction_count += other.correct_prediction_count


class EmbeddingAccuracy(Metric):

    def __init__(self, margin: float = 0.95) -> None:
        self.name = "embedding_accuracy"
        self.margin = margin
        self.outputs = []
        self.targets = []

    def __call__(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> float:
        output_similarity = output_batch @ output_batch.T
        target_similarity = (target_batch @ target_batch.T).type(torch.long)
        n_correct_predictions = (
            (
                (output_similarity > self.margin).type(torch.long)
                == target_similarity
            )
        ).sum().item()
        n_predictions = output_similarity.numel()
        return n_correct_predictions / n_predictions

    def update(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        self.outputs.extend(output_batch)
        self.targets.extend(target_batch)

    def result(self) -> float:
        outputs = torch.stack(self.outputs)
        targets = torch.stack(self.targets)
        return self(outputs, targets)

    def reset(self) -> None:
        self.outputs = []
        self.targets = []

    def merge(self, other) -> None:
        self.outputs.extend(other.outputs)
        self.targets.extend(other.targets)
