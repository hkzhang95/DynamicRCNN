import os
import argparse
import time
import datetime

import torch
import torch.distributed as dist

from config import config as cfg
from network import Network
from dataset import make_data_loader

from dynamic_rcnn.engine.comm import synchronize, get_rank, get_world_size
from dynamic_rcnn.utils.logger import setup_logger
from dynamic_rcnn.utils.metric_logger import MetricLogger
from dynamic_rcnn.utils.pyt_utils import mkdir
from dynamic_rcnn.engine.lr_scheduler import WarmupMultiStepLR
from dynamic_rcnn.engine.checkpoint import DetectronCheckpointer


def make_optimizer(cfg, model, scale_factor=1.):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR * scale_factor
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR * scale_factor
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Object Detection Training")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    num_gpus = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("train", output_dir, get_rank(),
                          filename='train_log.txt')
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    model = Network()
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # scaling policy, only suppose batch_size < SOLVER.IMS_PER_BATCH
    lr_steps, scale_factor = cfg.SOLVER.STEPS, 1.0
    batch_size = num_gpus * cfg.SOLVER.IMS_PER_GPU
    if batch_size < cfg.SOLVER.IMS_PER_BATCH:
        assert cfg.SOLVER.IMS_PER_BATCH % batch_size == 0
        scale_factor = cfg.SOLVER.IMS_PER_BATCH // batch_size
        lr_steps = [step * scale_factor for step in lr_steps]
    optimizer = make_optimizer(cfg, model, 1.0 / scale_factor)
    scheduler = WarmupMultiStepLR(
        optimizer, lr_steps, cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    checkpoint_dir = os.path.join(cfg.OUTPUT_DIR, 'checkpoints')
    mkdir(checkpoint_dir)

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, checkpoint_dir, save_to_disk, logger
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)
    start_iter = arguments["iteration"]

    data_loader = make_data_loader(
        num_gpus, is_train=True, is_distributed=args.distributed,
        start_iter=start_iter)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)

    model.train()
    start_training_time = time.time()
    end = time.time()

    rcnn_iou_now = cfg.MODEL.DYNAMIC_RCNN.WARMUP_IOU
    rcnn_beta_now = cfg.MODEL.DYNAMIC_RCNN.WARMUP_BETA
    iteration_count = cfg.MODEL.DYNAMIC_RCNN.ITERATION_COUNT
    S_I, S_E = [], []
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):

        if any(len(target) < 1 for target in targets):
            logger.error(
                "Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}")
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict, rcnn_iou_new, rcnn_error_new = model(
            images, targets, rcnn_iou=rcnn_iou_now, rcnn_beta=rcnn_beta_now)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        def reduce_loss_dict(loss_dict):
            """
            Reduce the loss dictionary from all processes so that process with rank
            0 has the averaged results. Returns a dict with the same fields as
            loss_dict, after reduction.
            """
            world_size = get_world_size()
            if world_size < 2:
                return loss_dict
            with torch.no_grad():
                loss_names = []
                all_losses = []
                for k in sorted(loss_dict.keys()):
                    loss_names.append(k)
                    all_losses.append(loss_dict[k])
                all_losses = torch.stack(all_losses, dim=0)
                dist.reduce(all_losses, dst=0)
                if dist.get_rank() == 0:
                    # only main process gets accumulated, so only divide by
                    # world_size in this case
                    all_losses /= world_size
                reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
            return reduced_losses

        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        S_I.append(rcnn_iou_new)
        S_E.append(rcnn_error_new)
        if iteration % iteration_count == 0:
            rcnn_iou_now = max(sum(S_I) / iteration_count,
                               cfg.MODEL.DYNAMIC_RCNN.WARMUP_IOU)
            rcnn_beta_now = min(sorted(S_E)[iteration_count // 2],
                                cfg.MODEL.DYNAMIC_RCNN.WARMUP_BETA)
            S_I, S_E = [], []

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0 or iteration == max_iter:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / max_iter
        )
    )


if __name__ == "__main__":
    main()
