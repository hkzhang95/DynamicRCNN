import os, getpass
import os.path as osp
import sys
import argparse
from easydict import EasyDict as edict


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


root_dir = osp.abspath(
    osp.join(osp.dirname(__file__), '..', '..', '..', '..', '..'))
add_path(root_dir)


class Config:
    user = getpass.getuser()
    this_model_dir = osp.split(os.path.realpath(__file__))[0]
    data_dir = osp.join(root_dir, 'data')
    OUTPUT_DIR = osp.join(
        root_dir, 'output', user, 'dynamic_rcnn', 'coco',
        os.path.split(os.path.split(os.path.realpath(__file__))[0])[1])

    # dataset setting
    DATASETS = edict()
    DATASETS.TRAIN = {
        "coco_2017_train": {
            "img_dir": osp.join(data_dir, "coco/train2017"),
            "ann_file": osp.join(
                data_dir, "coco/annotations/instances_train2017.json")}}
    DATASETS.TEST = {
        'coco_2017_val': {
            "img_dir": osp.join(data_dir, "coco/val2017"),
            "ann_file": osp.join(
                data_dir, "coco/annotations/instances_val2017.json")}}

    # input setting
    INPUT = edict()
    INPUT.MIN_SIZE_TRAIN = (400, 600, 800, 1000, 1200)
    INPUT.MAX_SIZE_TRAIN = 1400
    INPUT.MIN_SIZE_TEST = 800
    INPUT.MAX_SIZE_TEST = 1333
    INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
    INPUT.PIXEL_STD = [1.0, 1.0, 1.0]
    INPUT.TO_BGR255 = True
    INPUT.BRIGHTNESS = 0.0
    INPUT.CONTRAST = 0.0
    INPUT.SATURATION = 0.0
    INPUT.HUE = 0.0
    INPUT.HORIZONTAL_FLIP_PROB_TRAIN = 0.5
    INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0

    # data loader
    DATALOADER = edict()
    DATALOADER.ASPECT_RATIO_GROUPING = True
    DATALOADER.NUM_WORKERS = 4
    DATALOADER.SIZE_DIVISIBILITY = 32

    # training config
    # TODO: multi machine
    SOLVER = edict()
    SOLVER.BASE_LR = 0.02
    SOLVER.BIAS_LR_FACTOR = 2
    SOLVER.IMS_PER_GPU = 2
    SOLVER.IMS_PER_BATCH = 16
    SOLVER.GAMMA = 0.1
    SOLVER.MOMENTUM = 0.9
    SOLVER.WEIGHT_DECAY = 0.0001
    SOLVER.WEIGHT_DECAY_BIAS = 0
    SOLVER.CHECKPOINT_PERIOD = 2500
    SOLVER.MAX_ITER = 270000
    SOLVER.STEPS = [180000, 240000]
    SOLVER.WARMUP_FACTOR = 1.0 / 3
    SOLVER.WARMUP_ITERS = 500
    SOLVER.WARMUP_METHOD = 'linear'

    # test config
    TEST = edict()
    TEST.DETECTIONS_PER_IMG = 100
    TEST.EXPECTED_RESULTS = []
    TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
    TEST.IMS_PER_GPU = 1
    TEST.IMS_PER_BATCH = 8
    TEST.VIS_THRESH = 0.3

    # test-time augmentation
    TEST.BBOX_AUG = edict()
    TEST.BBOX_AUG.ENABLED = False
    TEST.BBOX_AUG.H_FLIP = False
    TEST.BBOX_AUG.SCALES = ()
    TEST.BBOX_AUG.MAX_SIZE = 4000
    TEST.BBOX_AUG.SCALE_H_FLIP = False

    # model config
    MODEL = edict()
    MODEL.DEVICE = 'cuda'
    MODEL.RPN_ONLY = False
    MODEL.MASK_ON = False
    MODEL.KEYPOINT_ON = False
    MODEL.CLS_AGNOSTIC_BBOX_REG = False
    MODEL.WEIGHT = osp.join(data_dir, 'basemodels', 'R-101.pkl')

    MODEL.RPN = edict()
    MODEL.RPN.USE_FPN = True
    MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
    MODEL.RPN.ANCHOR_STRIDE = (4, 8, 16, 32, 64)
    MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
    MODEL.RPN.STRADDLE_THRESH = 0
    MODEL.RPN.FG_IOU_THRESHOLD = 0.7
    MODEL.RPN.BG_IOU_THRESHOLD = 0.3
    MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    MODEL.RPN.POSITIVE_FRACTION = 0.5
    MODEL.RPN.NMS_THRESH = 0.85
    MODEL.RPN.MIN_SIZE = 0
    # per stage level
    MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 2000
    MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
    MODEL.RPN.PRE_NMS_TOP_N_TEST = 1000
    MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
    # per batch setting in Detectron
    MODEL.RPN.FPN_POST_NMS_PER_BATCH = True
    MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 1000 * SOLVER.IMS_PER_GPU
    MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 1000 * TEST.IMS_PER_GPU
    # # per image setting, maskrcnn-benchmark issue #672
    # MODEL.RPN.FPN_POST_NMS_PER_BATCH = False
    # MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 1000
    # MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 1000

    MODEL.FPN = edict()
    MODEL.FPN.USE_RELU = False

    MODEL.BACKBONE = edict()
    MODEL.BACKBONE.CONV_BODY = 'R-101-FPN'
    MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2

    MODEL.RESNETS = edict()
    MODEL.RESNETS.NUM_GROUPS = 1
    MODEL.RESNETS.WIDTH_PER_GROUP = 64
    MODEL.RESNETS.STRIDE_IN_1X1 = True
    MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
    MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"
    MODEL.RESNETS.RES5_DILATION = 1
    MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256
    MODEL.RESNETS.RES2_OUT_CHANNELS = 256
    MODEL.RESNETS.STEM_OUT_CHANNELS = 64
    MODEL.RESNETS.STAGE_WITH_DCN = (False, False, False, False)
    MODEL.RESNETS.WITH_MODULATED_DCN = False
    MODEL.RESNETS.DEFORMABLE_GROUPS = 1

    MODEL.ROI_HEADS = edict()
    MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
    MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
    MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100
    MODEL.ROI_HEADS.NMS = 0.5
    MODEL.ROI_HEADS.SCORE_THRESH = 0.05
    MODEL.ROI_HEADS.USE_FPN = True

    MODEL.ROI_BOX_HEAD = edict()
    MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024
    MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81
    MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2
    MODEL.ROI_BOX_HEAD.POOLER_SCALES = (0.25, 0.125, 0.0625, 0.03125)

    MODEL.DYNAMIC_RCNN = edict()
    MODEL.DYNAMIC_RCNN.KI = 75
    MODEL.DYNAMIC_RCNN.KE = 10
    MODEL.DYNAMIC_RCNN.ITERATION_COUNT = 100
    MODEL.DYNAMIC_RCNN.WARMUP_IOU = 0.4
    MODEL.DYNAMIC_RCNN.WARMUP_BETA = 1.0


config = Config()


def link_log_dir():
    if not os.path.exists(osp.join(config.this_model_dir, 'log')):
        cmd = "ln -s " + config.OUTPUT_DIR + " log"
        os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-log', '--linklog', default=False, action='store_true')
    args = parser.parse_args()
    if args.linklog:
        link_log_dir()
