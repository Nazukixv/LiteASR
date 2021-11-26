"""Initialize sub package."""

import importlib
import os

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from liteasr.config import LiteasrDataclass

OPTIMIZER_REGISTRY = {}
OPTIMIZER_DATACLASS_REGISTRY = {}
OPTIMIZER_CLASS_NAMES = set()


class LiteasrOptimizer(object):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def build_optimizer(cls, params, cfg, task):
        raise NotImplementedError

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.param_groups:
            for p in param_group["params"]:
                yield p

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.params:
            p.grad = None
        self.optimizer.zero_grad()

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string


def build_optimizer(params, cfg, task) -> LiteasrOptimizer:
    optim_name = getattr(cfg, "name", None)

    optim = OPTIMIZER_REGISTRY[optim_name]

    dc = OPTIMIZER_DATACLASS_REGISTRY[optim_name]
    merged_cfg = OmegaConf.merge(dc, cfg)
    merged_cfg.__dict__["_parent"] = cfg.__dict__["_parent"]
    OmegaConf.set_struct(merged_cfg, True)

    return optim.build_optimizer(params, merged_cfg, task)


def register_optimzer(name, dataclass=None):

    def register_optimizer_cls(cls):
        OPTIMIZER_REGISTRY[name] = cls
        OPTIMIZER_CLASS_NAMES.add(cls.__name__)

        if dataclass is not None:
            assert issubclass(dataclass, LiteasrDataclass)
            OPTIMIZER_DATACLASS_REGISTRY[name] = dataclass
            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="optimizer", node=node)

        return cls

    return register_optimizer_cls


# automatically import any Python files in the tasks/ directory
optims_dir = os.path.dirname(__file__)
for file in os.listdir(optims_dir):
    path = os.path.join(optims_dir, file)
    if (
        not file.startswith("_") and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        optim_name = file[:file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("liteasr.optims." + optim_name)
