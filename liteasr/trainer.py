"""Trainer."""

import logging
import math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from liteasr.config import LiteasrConfig
from liteasr.criterions import LiteasrLoss
from liteasr.distributed.ddp_model_wrapper import DDPModelWrapper
from liteasr.models import LiteasrModel
from liteasr.optims import LiteasrOptimizer
from liteasr.tasks import LiteasrTask
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
        self.epoch = 1000000

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

    def run(self):
        sampler = DistributedSampler(self.task.dataset)
        train_iter = DataLoader(
            dataset=self.task.dataset,
            batch_size=1,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=lambda x: x[0],
        )
        device = torch.device("cuda")

        for ep in range(self.epoch):
            train_iter.sampler.set_epoch(ep)
            for batch in train_iter:
                batch = to_device(batch, device)

                loss = self.criterion(self.model, *batch)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 5.0
                )
                if not math.isnan(grad_norm):
                    self.optimizer.step()
                self.optimizer.zero_grad()

            if ep % 100 == 0 and ep != 0:
                logger.info(
                    "epoch {} - current loss: {:.2f}".format(
                        ep,
                        loss.item(),
                    )
                )

            if ep % 1000 == 0 and ep != 0:
                if dist.get_rank() == 0:
                    self.model.eval()
                    with torch.no_grad():
                        i = 0
                        for data_batch in self.task.data:
                            for test_data in data_batch:
                                feats = test_data.x.unsqueeze(0).to(
                                    device=device
                                )
                                tgt = "".join(
                                    self.task.vocab.lookup(test_data.tokenids)
                                )
                                pred = self.task.inference(
                                    feats, model=self.model
                                )
                                logger.info(
                                    f"{'[X]' if tgt == pred else '[ ]'} {pred}"
                                )
                                i += 1
                    self.model.train()
