"""Trainer."""

import logging
import math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from liteasr.config import LiteasrConfig
from liteasr.criterions import LiteasrLoss
from liteasr.distributed.ddp_model_wrapper import DDPModelWrapper
from liteasr.models import LiteasrModel
from liteasr.optims import LiteasrOptimizer
from liteasr.tasks import LiteasrTask
from liteasr.utils.data_loader import EpochDataLoader
from liteasr.utils.device import to_device

logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(
        self,
        cfg: LiteasrConfig,
        task: LiteasrTask,
        model: LiteasrModel,
        criterion: LiteasrLoss,
        optimizer: LiteasrOptimizer,
    ):
        self.cfg = cfg
        self.task = task
        self._model = model
        self._wrapped_model = None
        self.criterion = criterion
        self.optimizer = optimizer
        self.update = 0

        if self.cfg.distributed.world_size > 1:
            sampler = DistributedSampler(self.task.dataset)
        else:
            sampler = None

        self.train_iter = EpochDataLoader(
            dataset=self.task.dataset,
            batch_size=1,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=lambda x: x[0],
        )

        self.device = torch.device("cuda")

        self.triggers = {
            "loss": Trigger(100, "iteration"),
            "valid": Trigger(2000, "iteration"),
        }

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.cfg.distributed.world_size > 1:
                ddp_model = DistributedDataParallel(
                    module=self._model,
                    device_ids=[self.cfg.distributed.device_id],
                    output_device=self.cfg.distributed.rank,
                )
                self._wrapped_model = DDPModelWrapper(ddp_model)
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def epoch(self):
        return self.train_iter.epoch

    def is_master(self):
        if self.cfg.distributed.world_size > 1:
            return dist.get_rank() == 0
        else:
            return True

    def stop(self):
        reach_max_epoch = (
            self.cfg.optimization.max_epoch >= 0
            and self.epoch >= self.cfg.optimization.max_epoch
        )
        reach_max_update = (
            self.cfg.optimization.max_update >= 0
            and self.update >= self.cfg.optimization.max_update
        )
        return reach_max_epoch or reach_max_update

    def run(self):
        for batch in self.train_iter:
            if self.stop():
                break
            batch = to_device(batch, self.device)

            loss = self.criterion(self.model, *batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 5.0
            )
            if not math.isnan(grad_norm):
                self.optimizer.step()
                self.update += 1
            self.optimizer.zero_grad()

            if self.triggers["loss"](self):
                logger.info(
                    "epoch {} - current loss: {:.2f}".format(
                        self.epoch,
                        loss.item(),
                    )
                )

            if self.triggers["valid"](self):
                if self.is_master():
                    self.model.eval()
                    with torch.no_grad():
                        # TODO: validation
                        pass
                    self.model.train()


class Trigger(object):

    def __init__(self, interval: int, unit: str):
        assert unit in ["epoch", "iteration"]
        self.interval = interval
        self.unit = unit
        self.prev_unit = 0

    def __call__(self, trainer: Trainer):
        if self.unit == "epoch":
            if trainer.epoch == self.prev_unit + self.interval:
                self.prev_unit += self.interval
                return True
            else:
                return False
        else:
            if trainer.update == self.prev_unit + self.interval:
                self.prev_unit += self.interval
                return True
            else:
                return False
