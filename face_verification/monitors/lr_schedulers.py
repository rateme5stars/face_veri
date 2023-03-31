from abc import ABCMeta, abstractmethod

import torch

from .monitors import Monitor, Criterion, Phase


class LRScheduler(Monitor, metaclass=ABCMeta):

    def __init__(
        self, scheduler, criterion: Criterion, phase: Phase = "valid"
    ) -> None:
        self.scheduler = scheduler
        self.criterion = criterion
        self.phase = phase

    def update(self, phase: Phase, model) -> None:
        if phase == self.phase:
            self.step()

    @abstractmethod
    def step(self) -> None:
        pass


class ReduceLROnPlateau(LRScheduler):

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        criterion: Criterion,
        phase: Phase = "valid",
    ) -> None:
        super().__init__(scheduler, criterion, phase)

    def step(self) -> None:
        self.scheduler.step(self.criterion.result())
