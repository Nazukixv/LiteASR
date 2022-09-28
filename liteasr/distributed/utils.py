import logging
import socket

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from liteasr.config import DistributedConfig
from liteasr.config import LiteasrConfig

logger = logging.getLogger(__name__)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def check_distributed_config(cfg: DistributedConfig):
    device_count = torch.cuda.device_count()
    if cfg.world_size > device_count:
        logger.warning(
            f"world_size changed from {cfg.world_size} -> {device_count}"
        )
        cfg.world_size = device_count


def infer_init_method(cfg: DistributedConfig):
    if cfg.init_method is None:
        sock = socket.socket()
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        cfg.init_method = "tcp://localhost:{}".format(port)


def distributed_init(cfg: LiteasrConfig):
    logger.info(
        "distributed init (rank {}) at {}".format(
            cfg.distributed.rank,
            cfg.distributed.init_method,
        )
    )
    dist.init_process_group(
        backend=cfg.distributed.backend,
        init_method=cfg.distributed.init_method,
        world_size=cfg.distributed.world_size,
        rank=cfg.distributed.rank,
    )
    logger.info(
        "initialized host {} as rank {}".format(
            socket.gethostname(),
            cfg.distributed.rank,
        )
    )
    # perform a dummy all-reduce to initialize the NCCL communicator
    if torch.cuda.is_available():
        dist.all_reduce(torch.zeros(1).cuda())

    cfg.distributed.rank = torch.distributed.get_rank()

    if cfg.distributed.rank != 0:
        logging.getLogger().setLevel(logging.WARNING)


def distributed_func(rank, func, cfg: LiteasrConfig):
    # set root logger config
    logging.config.dictConfig(
        OmegaConf.to_container(cfg.job_logging_cfg, resolve=True)
    )

    cfg.distributed.device_id = rank
    torch.cuda.set_device(cfg.distributed.device_id)

    cfg.distributed.rank = sum(
        cfg.distributed.world_piece_size[:cfg.distributed.machine_rank]
    ) + rank

    distributed_init(cfg)

    func(cfg)

    dist.destroy_process_group()


def call_func(func, cfg: LiteasrConfig):
    """Distributed function calling wrapper.

    Distributed training will use GPU as many as possible.
    """

    if not torch.cuda.is_available():
        logger.warning("CUDA is NOT available!")
        return
    elif torch.cuda.device_count() == 1 or cfg.distributed.world_size == 1:
        logger.info("using only one single GPU, not apply DDP training")
        func(cfg)
    else:
        # check_distributed_config(cfg.distributed)
        infer_init_method(cfg.distributed)
        mp.spawn(
            fn=distributed_func,
            args=(func, cfg),
            nprocs=(
                cfg.distributed.world_piece_size[cfg.distributed.machine_rank]
            ),
            join=True,
        )
