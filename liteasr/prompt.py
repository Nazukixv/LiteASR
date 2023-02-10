import argparse

from omegaconf import OmegaConf

from liteasr.models import MODEL_DATACLASS_REGISTRY
from liteasr.optims import OPTIMIZER_DATACLASS_REGISTRY
from liteasr.tasks import TASK_DATACLASS_REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("option", type=str, help="<group>.<option> e.g. task.asr")

    args = parser.parse_args()
    group, option = args.option.split(".")

    if group == "model":
        dc = MODEL_DATACLASS_REGISTRY[option]
    elif group == "task":
        dc = TASK_DATACLASS_REGISTRY[option]
    elif group == "optimizer":
        dc = OPTIMIZER_DATACLASS_REGISTRY[option]
    else:
        raise ValueError(f"{group} is not a module")

    dc.name = option
    print(OmegaConf.to_yaml(dc))


if __name__ == "__main__":
    main()
