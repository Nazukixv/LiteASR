"""DistributedDataParallel module wrapper."""

import torch.nn as nn

from liteasr.models import LiteasrModel


class DDPModelWrapper(LiteasrModel):
    def __init__(self, ddp_module: nn.Module):
        super().__init__()
        assert hasattr(
            ddp_module, "module"
        ), "DDPModelWrapper expects input to wrap another module"

        self.ddp_module = ddp_module

    def __getattr__(self, name):
        """Forward missing attributes to twice-wrapped module."""
        try:
            # defer to nn.Module's logic
            return super().__getattr__(name)
        except AttributeError:
            try:
                # forward to the once-wrapped module
                return getattr(self.ddp_module, name)
            except AttributeError:
                # forward to the twice-wrapped module
                return getattr(self.ddp_module.module, name)

    def state_dict(self, *args, **kwargs):
        """Forward to the twice-wrapped module."""
        return self.ddp_module.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Forward to the twice-wrapped module."""
        return self.ddp_module.module.load_state_dict(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.ddp_module(*args, **kwargs)

    def inference(self, x):
        return self.ddp_module.module.inference(x)

    def save(self, model_path):
        self.ddp_module.module.save(model_path)

    def get_pred_len(self, xlens):
        return self.ddp_module.module.get_pred_len(xlens)

    def get_target(self, ys, ylens):
        return self.ddp_module.module.get_target(ys, ylens)

    def get_target_len(self, ylens):
        return self.ddp_module.module.get_target_len(ylens)

    def no_sync(self):
        return self.ddp_module.no_sync()
