"""Initialize sub package."""

import importlib
import os

from torch import Tensor

from liteasr.config import PostProcessConfig

TRANS_REGISTRY = {}


def register_transformation(name):

    def register_transformation_cls(cls):
        TRANS_REGISTRY[name] = cls
        return cls

    return register_transformation_cls


# automatically import any Python files in the utils/transform/ directory
trans_dir = os.path.dirname(__file__)
for file in os.listdir(trans_dir):
    path = os.path.join(trans_dir, file)
    if (
        not file.startswith("_") and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
        and not file.endswith("deprecated.py")
    ):
        trans_name = file[:file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(
            "liteasr.utils.transform." + trans_name
        )


class PostProcess(object):

    def __init__(self, cfg: PostProcessConfig):
        self.workflow = []
        for name in cfg.workflow:
            sub_cfg = getattr(cfg, name)
            self.workflow.append(TRANS_REGISTRY[name](sub_cfg))

    def __call__(self, x: Tensor) -> Tensor:
        for transformation in self.workflow:
            x = transformation(x)

        return x
