from abc import ABC, abstractmethod


class BaseLLM(ABC):

    @abstractmethod
    def reconfigure(self, config: dict):
        pass

    @abstractmethod
    def complete(self, text: str) -> str:
        pass
