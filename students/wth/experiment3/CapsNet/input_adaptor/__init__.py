from abc import ABCMeta, abstractmethod


class InputAdaptor(metaclass=ABCMeta):

    @property
    @abstractmethod
    def image_width(self):
        pass

    @property
    @abstractmethod
    def image_height(self):
        pass

    @property
    @abstractmethod
    def label_length(self):
        pass

    @property
    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def batch(self, n):
        pass
