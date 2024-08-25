from abc import ABC, abstractmethod, ABCMeta
import torch

class Summarizer(ABC):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def summarize(self, *args, **kwargs):
        pass

class SingletonABCMeta(ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonABCMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
