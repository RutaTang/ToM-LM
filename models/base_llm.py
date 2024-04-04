from abc import ABC, abstractmethod


class BaseLLM(ABC):

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def reconfigure(self, config: dict):
        pass

    @abstractmethod
    def complete(self, text: str) -> str:
        pass
