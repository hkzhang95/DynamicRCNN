import argparse
import os
from tqdm import tqdm
import logging
import time
import datetime
import cv2
import numpy as np
import torch

from config import config as cfg
from network import Network
from dataset import make_data_loader

from dynamic_rcnn.datasets.coco import COCODataset
from dynamic_rcnn.engine.checkpoint import DetectronCheckpointer
from dynamic_rcnn.engine.comm import synchronize, get_rank, get_world_size, \
    all_gather, is_main_process
from dynamic_rcnn.utils.logger import setup_logger
from dynamic_rcnn.utils.pyt_utils import mkdir, draw_box
from dynamic_rcnn.engine.bbox_aug import im_detect_bbox_aug
from dynamic_rcnn.datasets.evaluation import evaluate


def compute_on_dataset(
        model, data_loader, device, bbox_aug, timer=None, show_res=False):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        if show_res:
            ori_imgs, ori_target, images, targets, image_ids = batch
        else:
            images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(cfg, model, images, device)
            else:
                output = model(images.to(device))
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        if show_res:
            for ori_img, pred_boxes, gt_boxes in \
                    zip(ori_imgs, output, targets):
                ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
                ori_shape = ori_img.shape[:2][::-1]
                pred_boxes = pred_boxes.resize(ori_shape)
                gt_boxes = gt_boxes.resize(ori_shape)
                # draw predictions
                for i in range(len(pred_boxes.bbox)):
                    color = (0, 255, 0)
                    score = float(pred_boxes.get_field('scores')[i])
                    if score > cfg.TEST.VIS_THRESH:
                        label_id = int(pred_boxes.get_field('labels')[i])
                        label = COCODataset.class_names[label_id]
                        draw_box(ori_img, pred_boxes.bbox[i], label, color,
                                 score=score)
                # draw ground-truths
                for i in range(len(gt_boxes.bbox)):
                    color = (0, 0, 255)
                    label_id = int(gt_boxes.get_field('labels')[i])
                    label = COCODataset.class_names[label_id]
                    draw_box(ori_img, gt_boxes.bbox[i], label, color)
                cv2.imshow('result', ori_img)
                while True:
                    c = cv2.waitKey(100000)
                    if c == ord('n'):
                        break
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("test.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        show_res=False,
        logger=None
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    if not logger:
        logger = logging.getLogger("test.inference")
    dataset = data_loader.dataset
    logger.info(
        "Start evaluation on {} dataset({} images).".format(dataset_name,
                                                            len(dataset)))

    start_time = time.time()
    predictions = compute_on_dataset(
        model, data_loader, device, bbox_aug=bbox_aug, show_res=show_res)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str,
            total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset, predictions=predictions,
                    output_folder=output_folder, logger=logger, **extra_args)


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Object Detection Testing")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--iter", "-i", type=int, default=-1,
                        help="The iteration number, default -1 which will "
                             "test the latest model")
    parser.add_argument('--show_res', '-s', default=False, action='store_true')

    args = parser.parse_args()

    num_gpus = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if args.show_res and num_gpus > 1:
        print('\033[93m You can\'t specify both show_image (-s) and multiple'
              ' devices (-d %s) \033[0m' % num_gpus)
        exit(-1)

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("test.inference", output_dir, get_rank(),
                          filename='test_log.txt')
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    model = Network()
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    checkpoint_dir = os.path.join(cfg.OUTPUT_DIR, 'checkpoints')
    mkdir(checkpoint_dir)

    checkpointer = DetectronCheckpointer(
        cfg, model, save_dir=checkpoint_dir, logger=logger)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    for idx, dataset_name in enumerate(dataset_names):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)
        output_folders[idx] = output_folder

    data_loaders = make_data_loader(
        num_gpus, is_train=False, is_distributed=distributed,
        return_raw=args.show_res)

    def test_model(model):
        for output_folder, dataset_name, data_loader_val in zip(
                output_folders, dataset_names, data_loaders):
            inference(
                model, data_loader_val, dataset_name=dataset_name,
                iou_types=iou_types, device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder, box_only=False,
                bbox_aug=cfg.TEST.BBOX_AUG.ENABLED, show_res=args.show_res,
                logger=logger)
            synchronize()

    test_iter = args.iter
    if args.iter == -1:
        model_file = os.readlink(
            os.path.join(checkpoint_dir, 'last_checkpoint'))
        test_iter = int(model_file.split('/')[-1].split('_')[-1][:-4])
    else:
        model_file = os.path.join(checkpoint_dir,
                                  "model_{:07d}.pth".format(args.iter))

    if os.path.exists(model_file):
        logger.info(
            "\n\nstart to evaluate iteration of {}".format(test_iter))
        _ = checkpointer.load(model_file)
        test_model(model)


if __name__ == "__main__":
    main()
