import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config as cfg
from dynamic_rcnn.basemodels import resnet
from dynamic_rcnn.det_opr.box_coder import BoxCoder
from dynamic_rcnn.det_opr.fpn.fpn import build_resnet_fpn_backbone
from dynamic_rcnn.det_opr.rpn.anchor_generator import make_anchor_generator
from dynamic_rcnn.det_opr.rpn.anchor_target_opr import anchor_target_opr
from dynamic_rcnn.det_opr.rpn.proposal_opr import proposal_opr
from dynamic_rcnn.det_opr.rcnn.proposal_target_opr import proposal_target_opr
from dynamic_rcnn.det_opr.rcnn.post_processing import post_processing_opr
from dynamic_rcnn.det_opr.poolers import make_pooler
from dynamic_rcnn.det_opr.loss import smooth_l1_loss
from dynamic_rcnn.datasets.structures.image_list import to_image_list
from dynamic_rcnn.utils.torch_utils import cat, concat_box_prediction_layers


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # backbone
        basemodel = resnet.ResNet(cfg)
        self.backbone = build_resnet_fpn_backbone(basemodel, cfg)

        # rpn
        self.anchor_generator = make_anchor_generator(cfg)
        rpn_in_channels = self.backbone.out_channels
        num_anchors = self.anchor_generator.num_anchors_per_location()[0]
        self.rpn_conv = nn.Conv2d(rpn_in_channels, rpn_in_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.rpn_cls_logits = nn.Conv2d(rpn_in_channels, num_anchors,
                                        kernel_size=1, stride=1)
        self.rpn_bbox_pred = nn.Conv2d(rpn_in_channels, num_anchors * 4,
                                       kernel_size=1, stride=1)
        self.rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # rcnn
        self.rcnn_roi_pooler = make_pooler(cfg, 'ROI_BOX_HEAD')
        rcnn_in_channels = self.backbone.out_channels
        input_size = rcnn_in_channels * \
                     cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = nn.Linear(input_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.cls_score = nn.Linear(
            representation_size, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else \
            cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.bbox_pred = nn.Linear(
            representation_size, num_bbox_reg_classes * 4)
        self.rcnn_box_coder = BoxCoder(
            weights=cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS)

        self._init_weights()

    def _init_weights(self):
        for l in [self.rpn_conv, self.rpn_cls_logits,
                  self.rpn_bbox_pred, self.cls_score]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        for fc in [self.fc6, self.fc7]:
            nn.init.kaiming_uniform_(fc.weight, a=1)
            nn.init.constant_(fc.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, images, targets=None,
                rcnn_iou=cfg.MODEL.DYNAMIC_RCNN.WARMUP_IOU,
                rcnn_beta=cfg.MODEL.DYNAMIC_RCNN.WARMUP_BETA):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)

        # rpn
        anchors = self.anchor_generator(images, features)
        rpn_cls_logits = []
        rpn_bbox_preds = []
        for feature in features:
            t = F.relu(self.rpn_conv(feature))
            rpn_cls_logits.append(self.rpn_cls_logits(t))
            rpn_bbox_preds.append(self.rpn_bbox_pred(t))

        # generate proposals
        pre_nms_top_n = cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN if \
            self.training else cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = cfg.MODEL.RPN.POST_NMS_TOP_N_TRAIN if \
            self.training else cfg.MODEL.RPN.POST_NMS_TOP_N_TEST
        # assert cfg.MODEL.RPN.FPN_POST_NMS_PER_BATCH == False
        fpn_post_nms_top_n = cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN if \
            self.training else cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST
        with torch.no_grad():
            proposals = proposal_opr(
                anchors, rpn_cls_logits, rpn_bbox_preds, self.rpn_box_coder,
                pre_nms_top_n, post_nms_top_n, cfg.MODEL.RPN.NMS_THRESH,
                cfg.MODEL.RPN.MIN_SIZE, fpn_post_nms_top_n=fpn_post_nms_top_n,
                fpn_post_nms_per_batch=cfg.MODEL.RPN.FPN_POST_NMS_PER_BATCH,
                is_train=self.training, targets=targets)

        # loss
        if self.training:
            # rpn loss
            labels, regression_targets, sampled_inds, sampled_pos_inds = \
                anchor_target_opr(
                    anchors, targets, self.rpn_box_coder,
                    cfg.MODEL.RPN.FG_IOU_THRESHOLD,
                    cfg.MODEL.RPN.BG_IOU_THRESHOLD,
                    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
                    cfg.MODEL.RPN.POSITIVE_FRACTION)

            objectness, box_regression = \
                concat_box_prediction_layers(rpn_cls_logits, rpn_bbox_preds)
            objectness = objectness.squeeze()

            rpn_bbox_loss = smooth_l1_loss(
                box_regression[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                beta=1.0 / 9,
                size_average=False,
            ) / (sampled_inds.numel())

            rpn_cls_loss = F.binary_cross_entropy_with_logits(
                objectness[sampled_inds], labels[sampled_inds]
            )

            # training proposals
            with torch.no_grad():
                proposals, raw_proposals = proposal_target_opr(
                    proposals, targets, self.rcnn_box_coder,
                    rcnn_iou, rcnn_iou,
                    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
                    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
                    return_ious=True, return_raw_proposals=True)

            # rcnn head
            rcnn_features = self.rcnn_roi_pooler(features, proposals)
            rcnn_features = rcnn_features.view(rcnn_features.size(0), -1)
            rcnn_features = F.relu(self.fc6(rcnn_features))
            rcnn_features = F.relu(self.fc7(rcnn_features))
            if rcnn_features.ndimension() == 4:
                assert list(rcnn_features.shape[2:]) == [1, 1]
                rcnn_features = rcnn_features.view(rcnn_features.size(0), -1)
            class_logits = self.cls_score(rcnn_features)
            box_regression = self.bbox_pred(rcnn_features)

            class_logits = cat([class_logits], dim=0)
            box_regression = cat([box_regression], dim=0)
            device = class_logits.device

            labels = cat(
                [proposal.get_field("labels") for proposal in proposals],
                dim=0)
            regression_targets = cat(
                [proposal.get_field("regression_targets") for proposal in
                 proposals], dim=0
            )
            # record the statistics
            ious = cat(
                [proposal.get_field("ious") for proposal in raw_proposals],
                dim=0)
            rcnn_iou_new = torch.topk(ious, cfg.MODEL.DYNAMIC_RCNN.KI *
                                      cfg.SOLVER.IMS_PER_GPU)[0][-1].item()
            raw_regression_targets = cat(
                [proposal.get_field("regression_targets") for proposal in
                 raw_proposals], dim=0
            ).abs()[:, :2].mean(dim=1)
            rcnn_error_new = torch.kthvalue(raw_regression_targets.cpu(), min(
                cfg.MODEL.DYNAMIC_RCNN.KE * cfg.SOLVER.IMS_PER_GPU,
                raw_regression_targets.size(0)))[0].item()

            rcnn_cls_loss = F.cross_entropy(class_logits, labels)

            # get indices that correspond to the regression targets for
            # the corresponding ground truth labels, to be used with
            # advanced indexing
            sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
            labels_pos = labels[sampled_pos_inds_subset]
            if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
                map_inds = torch.tensor([4, 5, 6, 7], device=device)
            else:
                map_inds = 4 * labels_pos[:, None] + torch.tensor(
                    [0, 1, 2, 3], device=device)

            box_loss = smooth_l1_loss(
                box_regression[sampled_pos_inds_subset[:, None], map_inds],
                regression_targets[sampled_pos_inds_subset],
                size_average=False,
                beta=rcnn_beta,
            )
            rcnn_bbox_loss = box_loss / labels.numel()

            losses = dict(rpn_cls_loss=rpn_cls_loss,
                          rpn_bbox_loss=rpn_bbox_loss,
                          rcnn_cls_loss=rcnn_cls_loss,
                          rcnn_bbox_loss=rcnn_bbox_loss)
            return losses, rcnn_iou_new, rcnn_error_new
        else:
            # rcnn head
            rcnn_features = self.rcnn_roi_pooler(features, proposals)
            rcnn_features = rcnn_features.view(rcnn_features.size(0), -1)
            rcnn_features = F.relu(self.fc6(rcnn_features))
            rcnn_features = F.relu(self.fc7(rcnn_features))
            if rcnn_features.ndimension() == 4:
                assert list(rcnn_features.shape[2:]) == [1, 1]
                rcnn_features = rcnn_features.view(rcnn_features.size(0),
                                                   -1)
            rcnn_cls_logits = self.cls_score(rcnn_features)
            rcnn_bbox_preds = self.bbox_pred(rcnn_features)

            # post processing
            results = post_processing_opr(
                proposals, rcnn_cls_logits, rcnn_bbox_preds,
                self.rcnn_box_coder,
                score_thresh=cfg.MODEL.ROI_HEADS.SCORE_THRESH,
                nms_thresh=cfg.MODEL.ROI_HEADS.NMS,
                detections_per_img=cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG,
                cls_agnostic_bbox_reg=cfg.MODEL.CLS_AGNOSTIC_BBOX_REG,
                bbox_aug_enabled=cfg.TEST.BBOX_AUG.ENABLED
            )

            return results
