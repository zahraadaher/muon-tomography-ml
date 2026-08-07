from abc import ABC, abstractmethod


class BaseDataModule(ABC):

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def create_datasets(self):
        pass

    @abstractmethod
    def get_metadata(self):
        pass