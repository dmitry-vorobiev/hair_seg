import datetime as dt
import hydra
import logging
import os
import time
import torch
import torch.distributed as dist
import torchvision

from hydra.utils import instantiate
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Metric, RunningAverage
from ignite.utils import convert_tensor
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from typing import Any, Dict, List, Optional, Tuple, Sized

from data.figaro1k import Dummy
from my_types import Batch, Device, FloatDict

Metrics = Dict[str, Metric]


def humanize_time(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')


def on_epoch_start(engine: Engine):
    engine.state.t0 = time.time()


def log_iter(engine, pbar, interval_steps=100):
    # type: (Engine, ProgressBar, Optional[int]) -> None
    epoch = engine.state.epoch
    iteration = engine.state.iteration
    metrics = engine.state.metrics
    stats = ", ".join(["%s: %.3f" % k_v for k_v in metrics.items()])
    t0 = engine.state.t0
    t1 = time.time()
    it_time = (t1 - t0) / interval_steps
    cur_time = humanize_time(t1)
    pbar.log_message("[{}][{:.3f} s] | ep: {:2d}, it: {:3d}, {}".format(
        cur_time, it_time, epoch, iteration, stats))
    engine.state.t0 = t1


def log_epoch(engine: Engine) -> None:
    epoch = engine.state.epoch
    metrics = engine.state.metrics
    stats = ", ".join(["%s: %.3f" % k_v for k_v in metrics.items()])
    logging.info("ep: {}, {}".format(epoch, stats))


def create_trainer(update_func, metrics=None):
    trainer = Engine(update_func)
    if metrics:
        for name, metric in metrics.items():
            metric.attach(trainer, name)
    return trainer


def _prepare_batch(batch: Batch, device: torch.device, non_blocking: bool) -> Batch:
    kwargs = dict(device=device, non_blocking=non_blocking)
    x, y = batch
    return convert_tensor(x, **kwargs), convert_tensor(y, **kwargs)


def create_metrics(keys: List[str], device: Device = None) -> Metrics:
    def _out_transform(kek: str):
        return lambda out: out[kek]

    metrics = {key: RunningAverage(output_transform=_out_transform(key),
                                   device=device)
               for key in keys}
    return metrics


def _upd_pbar_iter_from_cp(engine: Engine, pbar: ProgressBar) -> None:
    pbar.n = engine.state.iteration


def create_dataset(conf, transforms):
    # type: (DictConfig, DictConfig) -> ImageFolder
    transforms = T.Compose([instantiate(v) for k, v in transforms.items()])
    ds = Dummy(conf.root, transform=transforms)
    return ds


def create_train_loader(conf, rank=None, num_replicas=None):
    # type: (DictConfig, Optional[int], Optional[int]) -> Sized
    data = create_dataset(conf, conf.transforms)
    print("Found {} images".format(len(data)))

    sampler = None
    if num_replicas is not None:
        sampler = DistributedSampler(data, num_replicas=num_replicas, rank=rank)

    loader = DataLoader(data,
                        sampler=sampler,
                        batch_size=conf.loader.batch_size,
                        num_workers=conf.get('loader.workers', 0),
                        drop_last=True)
    return loader


def setup_checkpoints(trainer, obj_to_save, epoch_length, conf):
    # type: (Engine, Dict[str, Any], int, DictConfig) -> None
    cp = conf.checkpoints
    save_path = cp.get('save_dir', os.getcwd())
    logging.info("Saving checkpoints to {}".format(save_path))
    max_cp = max(int(cp.get('max_checkpoints', 1)), 1)
    save = DiskSaver(save_path, create_dir=True, require_empty=True)
    make_checkpoint = Checkpoint(obj_to_save, save, n_saved=max_cp)
    cp_iter = cp.interval_iteration
    cp_epoch = cp.interval_epoch
    if cp_iter > 0:
        save_event = Events.ITERATION_COMPLETED(every=cp_iter)
        trainer.add_event_handler(save_event, make_checkpoint)
    if cp_epoch > 0:
        if cp_iter < 1 or epoch_length % cp_iter:
            save_event = Events.EPOCH_COMPLETED(every=cp_epoch)
            trainer.add_event_handler(save_event, make_checkpoint)


def run(conf: DictConfig, local_rank=0, distributed=False):
    epochs = conf.train.epochs
    epoch_length = conf.train.epoch_length
    torch.manual_seed(conf.general.seed)

    if distributed:
        rank = dist.get_rank()
        num_replicas = dist.get_world_size()
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        num_replicas = 1
        torch.cuda.set_device(conf.general.gpu)
    device = torch.device('cuda')
    loader_args = dict()
    master_node = rank == 0

    if master_node:
        print(conf.pretty())
    if num_replicas > 1:
        epoch_length = epoch_length // num_replicas
        loader_args = dict(rank=rank, num_replicas=num_replicas)

    train_dl = create_train_loader(conf.data, **loader_args)

    if epoch_length < 1:
        epoch_length = len(train_dl)

    net: nn.Module = instantiate(conf.model).to(device)
    criterion = instantiate(conf.loss).to(device)
    optim: torch.optim.SGD = instantiate(conf.optim, net.parameters())

    to_save = {
        'net': net,
        'optim': optim,
    }

    if master_node and conf.logging.model:
        logging.info(net)

    if distributed:
        ddp_kwargs = dict(device_ids=[local_rank, ], output_device=local_rank)
        net = torch.nn.parallel.DistributedDataParallel(net, **ddp_kwargs)

    update_freq = conf.optim.step_interval

    def _update(eng: Engine, batch: Batch) -> FloatDict:
        it = eng.state.iteration
        net.train()
        x, y = _prepare_batch(batch, device, non_blocking=True)
        y_hat = net(x)
        loss, stats = criterion(x, y, y_hat)

        if not it % update_freq:
            optim.zero_grad()
        loss.backward()
        if not (it + 1) % update_freq:
            optim.step()

        return stats

    metric_names = getattr(criterion, "stats_names", lambda: [])()
    metrics = create_metrics(metric_names, device if distributed else None)

    trainer = create_trainer(_update, metrics)
    to_save['trainer'] = trainer

    every_iteration = Events.ITERATION_COMPLETED
    trainer.add_event_handler(every_iteration, TerminateOnNan())

    cp = conf.checkpoints
    pbar = None

    if master_node:
        log_freq = conf.logging.iter_freq
        log_event = Events.ITERATION_COMPLETED(every=log_freq)
        pbar = ProgressBar(persist=False)
        trainer.add_event_handler(Events.EPOCH_STARTED, on_epoch_start)
        trainer.add_event_handler(log_event, log_iter, pbar, log_freq)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_epoch)
        pbar.attach(trainer, metric_names=metric_names)
        setup_checkpoints(trainer, to_save, epoch_length, conf)

    if 'load' in cp.keys() and cp.load is not None:
        if master_node:
            logging.info("Resume from a checkpoint: {}".format(cp.load))
            trainer.add_event_handler(Events.STARTED, _upd_pbar_iter_from_cp, pbar)
        Checkpoint.load_objects(to_load=to_save,
                                checkpoint=torch.load(cp.load, map_location=device))

    try:
        trainer.run(train_dl, max_epochs=epochs, epoch_length=epoch_length)
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
    if pbar is not None:
        pbar.close()


@hydra.main(config_path="../config/train.yaml")
def main(conf: DictConfig):
    env = os.environ.copy()
    world_size = int(env.get('WORLD_SIZE', -1))
    local_rank = int(env.get('LOCAL_RANK', -1))
    dist_conf: DictConfig = conf.distributed
    distributed = world_size > 1 and local_rank >= 0

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Unable to find any CUDA device")

        torch.backends.cudnn.benchmark = True
        dist.init_process_group(dist_conf.backend, init_method=dist_conf.url)
        if local_rank == 0:
            print("\nDistributed setting:")
            print("\tbackend: {}".format(dist.get_backend()))
            print("\tworld size: {}".format(dist.get_world_size()))
            print("\trank: {}\n".format(dist.get_rank()))

    try:
        run(conf, local_rank, distributed)
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        if distributed:
            dist.destroy_process_group()
        raise e

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
