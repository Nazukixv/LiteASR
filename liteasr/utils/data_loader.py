"""Data loader."""

from torch.utils.data.dataloader import DataLoader


class EpochDataLoader(object):

    def __init__(self, **kwargs):
        self.data_loader = DataLoader(**kwargs)
        self.epoch = 0
        self.data_iter = None

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        while True:
            try:
                if self.data_iter is None:
                    self._init_data_iter(self.epoch)
                yield next(self.data_iter)
            except StopIteration:
                self.epoch += 1
                self._init_data_iter(self.epoch)
                yield next(self.data_iter)

    def _init_data_iter(self, epoch):
        if hasattr(self.data_loader.sampler, "set_epoch"):
            self.data_loader.sampler.set_epoch(epoch)
        self.data_iter = iter(self.data_loader)
