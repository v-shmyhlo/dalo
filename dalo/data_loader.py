import abc
import os
from multiprocessing import Pool


class DataLoader(abc.ABC):
    def __iter__(self):
        with Pool(os.cpu_count()) as pool:
            yield from pool.imap(self.load_sample, self.samples())

    @abc.abstractmethod
    def load_sample(self, sample):
        ...

    @abc.abstractmethod
    def samples(self):
        ...
