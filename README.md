# Dynamic R-CNN: Towards High Quality Object Detection via Dynamic Training

By [Hongkai Zhang](https://hkzhang95.github.io/), [Hong Chang](https://scholar.google.com/citations?user=LX6MnNsAAAAJ&hl=en), [Bingpeng Ma](http://people.ucas.edu.cn/~bpma), [Naiyan Wang](https://winsty.net/), [Xilin Chen](http://vipl.ict.ac.cn/en/people/~xlchen).

This project is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).

[2020.7] Dynamic R-CNN is officially included in [MMDetection V2.2](https://github.com/open-mmlab/mmdetection/tree/master/configs/dynamic_rcnn), many thanks to [@xvjiarui](https://github.com/xvjiarui) and [@hellock](https://github.com/hellock) for migrating the code.

## Abstract

Although two-stage object detectors have continuously advanced the state-of-the-art performance in recent years, the training process itself is far from crystal. In this work, we first point out the inconsistency problem between the fixed network settings and the dynamic training procedure, which greatly affects the performance. For example, the fixed label assignment strategy and regression loss function cannot fit the distribution change of proposals and are harmful to training high quality detectors. Then, we propose *Dynamic R-CNN* to adjust the label assignment criteria (IoU threshold) and the shape of regression loss function (parameters of SmoothL1 Loss) automatically based on the statistics of proposals during training. This dynamic design makes better use of the training samples and pushes the detector to fit more high quality samples. Specifically, our method improves upon ResNet-50-FPN baseline with 1.9% AP and 5.5% AP90 on the MS COCO dataset with no extra overhead. For more details, please refer to our [paper](https://arxiv.org/abs/2004.06002).

## Models

Model | Multi-scale training | AP (minival) | AP (test-dev) | Trained model
--- |:---:|:---:|:---:|:---:
Dynamic_RCNN_r50_fpn_1x | No | 38.9 | 39.1 | [Google Drive](https://drive.google.com/open?id=1vFKc3FIw26uMTY92cyME0hcHbr5f4MR1)
Dynamic_RCNN_r50_fpn_2x | No | 39.9 | 39.9 | [Google Drive](https://drive.google.com/open?id=1zHXIshC7qbK_Jn9BiribtaJ1pe_NZ6WL)
Dynamic_RCNN_r101_fpn_1x | No | 41.0 | 41.2 | [Google Drive](https://drive.google.com/open?id=1ARhu8Eynnbj1R4Oh-_mw1UZ4WsLg9kQK)
Dynamic_RCNN_r101_fpn_2x | No | 41.8 | 42.0 | [Google Drive](https://drive.google.com/open?id=16eS1W39hnYtwsOLoQ5xTrYxAvlB689Xi)
Dynamic_RCNN_r101_fpn_3x | Yes | 44.4 | 44.7 | [Google Drive](https://drive.google.com/open?id=19NxzuMBQf2H7MAzflhg6_TgkPgz4AvyC)
Dynamic_RCNN_r101_dcnv2_fpn_3x | Yes | 46.7 | 46.9 | [Google Drive](https://drive.google.com/open?id=1VGFsaPZRrQ4dSV8APDTaueCEzu4dKSVM)

1. `1x`, `2x` and `3x` mean the model is trained for 90K, 180K and 270K iterations, respectively.
2. For `Multi-scale training`, the shorter side of images is randomly chosen from (400, 600, 800, 1000, 1200), and the longer side is 1400. We also extend the training time by `1.5x` under this setting.
3. `dcnv2` denotes deformable convolutional networks v2. We follow the same setting as [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Note that the result of this version is slightly lower than that of [mmdetection](https://github.com/open-mmlab/mmdetection).
4. All results in the table are obtained using a single model with no extra testing tricks. Additionally, adopting multi-scale testing on model `Dynamic_RCNN_r101_dcnv2_fpn_3x` achieves 49.2% in AP on COCO test-dev. Please set `TEST.BBOX_AUG.ENABLED = True` in the `config.py` to enable multi-scale testing. Here we use five scales with shorter sides (800, 1000, 1200, 1400, 1600) and the longer side is 2000 pixels. Note that Dynamic R-CNN*(50.1% AP) in Table 9 is implemented using MMDetection v1.1, please refer to this [link](https://github.com/hkzhang95/DynamicRCNN-mmdetV1.1/blob/master/configs/dynamic_rcnn).
5. If you want to test the model provided by us, please refer to [Testing](#Testing).

## Getting started

### Installation
#### 0. Requirements
- pytorch (v1.0.1.post2, other version have not been tested)
- torchvision (v0.2.2.post3, other version have not been tested)
- cocoapi
- matplotlib
- tqdm
- cython
- easydict
- opencv

Anaconda3 is recommended here since it integrates many useful packages. Please make sure that your conda is setup properly with the right environment. Then install `pytorch` and `torchvision` manually as follows:

```bash
pip install torch==1.0.1.post2
pip install torchvision==0.2.2.post3
```

Other dependencies will be installed during `setup`.

#### 1. Clone this repo

```bash
git clone https://github.com/hkzhang95/DynamicRCNN.git
```

#### 2. Compile kernels

Please make sure your `CUDA` is successfully installed and be added to the `PATH`. I only test `CUDA-9.0` for my experiments.

```bash
cd ${DynamicRCNN_ROOT}
python setup.py build develop
```

#### 3. Prepare data and output directory

```bash
cd ${DynamicRCNN_ROOT}
mkdir data
mkdir output
```

Prepare data and pretrained models:
- [COCO dataset](http://cocodataset.org/#download)
- [ImageNet Pretrained Models from Detectron](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#imagenet-pretrained-models)

Then organize them as follows:

```
DynamicRCNN
├── dynamic_rcnn
├── models
├── output
├── data
│   ├── basemodels/R-50.pkl
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017(2014)
│   │   ├── val2017(2014)
```

### Training

We use `torch.distributed.launch` in order to launch multi-gpu training.

```bash
cd models/zhanghongkai/dynamic_rcnn/coco/dynamic_rcnn_r50_fpn_1x
python -m torch.distributed.launch --nproc_per_node=8 train.py
```

### Outputs

Training and testing logs will be saved automatically in the `output` directory following the same path as in `models`.

For example, the experiment directory and log directory are formed as follows:

```
models/zhanghongkai/dynamic_rcnn/coco/dynamic_rcnn_r50_fpn_1x
output/zhanghongkai/dynamic_rcnn/coco/dynamic_rcnn_r50_fpn_1x
```

And you can link the `log` to your experiment directory by this script in the experiment directory:

```bash
python config.py -log
```

### Testing

Using `-i` to specify iteration for testing, default is the latest model.

```bash
# for regular testing and evaluation
python -m torch.distributed.launch --nproc_per_node=8 test.py
# for specified iteration
python -m torch.distributed.launch --nproc_per_node=8 test.py -i $iteration_number
```

If you want to test our provided model, just download the model, move it to the corresponding log directory and create a symbolic link like follows:

```bash
# example for Dynamic_RCNN_r50_fpn_1x
cd models/zhanghongkai/dynamic_rcnn/coco/dynamic_rcnn_r50_fpn_1x
python config.py -log
realpath log | xargs mkdir
mkdir -p log/checkpoints
mv path/to/the/model log/checkpoints
realpath log/checkpoints/dynamic_rcnn_r50_fpn_1x_test_model_0090000.pth last_checkpoint | xargs ln -s
```

Then you can follow the regular testing and evaluation process.

## Third-party resources

- MxNet implementation: [SimpleDet](https://github.com/TuSimple/simpledet)

## Acknowledgement

- [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
- [light_head_rcnn](https://github.com/zengarden/light_head_rcnn)

## Citations

Please consider citing our paper in your publications if it helps your research:

```
@inproceedings{DynamicRCNN,
    author = {Zhang, Hongkai and Chang, Hong and Ma, Bingpeng and Wang, Naiyan and Chen, Xilin},
    title = {Dynamic {R-CNN}: Towards High Quality Object Detection via Dynamic Training},
    booktitle = {ECCV},
    year = {2020}
}
```
