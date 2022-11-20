from abc import ABC, abstractmethod


class StoppingCriteria(ABC):
    @abstractmethod
    def __call__(self, loss: float) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class IterationStoppingCriteria(StoppingCriteria):
    def __init__(self, max_iter: int):
        self.max_iter = max_iter
        self.reset()

    def __call__(self, loss: float) -> bool:
        self.iter += 1
        return self.iter >= self.max_iter

    def reset(self) -> None:
        self.iter = 0


class NotImprovingStoppingCriteria(StoppingCriteria):
    def __init__(self, max_iter: int, eps: float = 0.0):
        self.max_iter = max_iter
        self.eps = eps
        self.reset()

    def __call__(self, loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = loss
            return False

        self.best_loss = min(self.best_loss, loss)

        if loss - self.best_loss > self.eps:
            self.iter += 1
        else:
            self.iter = 0

        return self.iter >= self.max_iter

    def reset(self) -> None:
        self.iter = 0
        self.best_loss = None
