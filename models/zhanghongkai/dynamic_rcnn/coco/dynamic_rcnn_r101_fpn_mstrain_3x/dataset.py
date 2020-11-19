from config import config as cfg

import torch.utils.data
from dynamic_rcnn.datasets.coco import COCODataset
from dynamic_rcnn.datasets.concat_dataset import ConcatDataset
from dynamic_rcnn.datasets.transforms import build_transforms
from dynamic_rcnn.datasets import samplers
from dynamic_rcnn.datasets.collate_batch import BatchCollator, BBoxAugCollator


def make_data_loader(
        num_gpus, is_train=True, is_distributed=False, start_iter=0,
        return_raw=False):
    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
        build_transforms(cfg, is_train)
    images_per_gpu = cfg.SOLVER.IMS_PER_GPU if is_train else cfg.TEST.IMS_PER_GPU
    images_per_batch = images_per_gpu * num_gpus

    if is_train:
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
        # scale, only suppose images_per_batch < SOLVER.IMS_PER_BATCH
        if images_per_batch < cfg.SOLVER.IMS_PER_BATCH:
            assert cfg.SOLVER.IMS_PER_BATCH % images_per_batch == 0
            num_iters *= (cfg.SOLVER.IMS_PER_BATCH // images_per_batch)
    else:
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    datasets = []
    for d_key, d_val in dataset_list.items():
        dataset = COCODataset(
            d_val['ann_file'], d_val['img_dir'],
            remove_images_without_annotations=is_train,
            transforms=transforms, return_raw=return_raw)
        datasets.append(dataset)
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    # make data sampler
    if is_distributed:
        sampler = samplers.DistributedSampler(dataset, shuffle=shuffle)
    elif shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)

    # make batch data sampler
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, dataset, aspect_grouping, images_per_gpu,
            drop_uneven=False)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_gpu, drop_last=False)

    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter)

    collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED \
        else BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY,
                           return_raw=return_raw)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=collator
    )
    if not is_train:
        data_loader = [data_loader]
    return data_loader
