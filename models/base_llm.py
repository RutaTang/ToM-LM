from abc import ABC, abstractmethod


class BaseLLM(ABC):

    @abstractmethod
    def complete(self, text: str) -> str:
        pass
