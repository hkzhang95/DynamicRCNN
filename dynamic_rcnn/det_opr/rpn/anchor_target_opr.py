# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from dynamic_rcnn.det_opr.matcher import Matcher
from dynamic_rcnn.det_opr.sampler import BalancedPositiveNegativeSampler
from dynamic_rcnn.datasets.structures.boxlist_ops import cat_boxlist, boxlist_iou


def anchor_target_opr(
        anchors, targets, box_coder, high_threshold, low_threshold,
        batch_size_per_image, positive_fraction):
    """
    Generate anchor targets for computing loss.

    Args:
        anchors: (list[BoxList])
        targets: (list[BoxList])
        box_coder: (BoxCoder)
        high_threshold: (float)
        low_threshold: (float)
        batch_size_per_image: (int)
        positive_fraction: (float)
    """
    matcher = Matcher(
        high_threshold, low_threshold, allow_low_quality_matches=True)
    fg_bg_sampler = BalancedPositiveNegativeSampler(
        batch_size_per_image, positive_fraction)

    anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
    # prepare targets
    labels = []
    regression_targets = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        # match targets to anchors
        match_quality_matrix = boxlist_iou(targets_per_image, anchors_per_image)
        matched_idxs = matcher(match_quality_matrix)
        targets_per_image = targets_per_image.copy_with_fields([])
        matched_targets = targets_per_image[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)

        matched_idxs = matched_targets.get_field("matched_idxs")
        # generate rpn labels
        labels_per_image = matched_idxs >= 0
        labels_per_image = labels_per_image.to(dtype=torch.float32)

        # Background (negative examples)
        bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
        labels_per_image[bg_indices] = 0

        # discard anchors that go out of the boundaries of the image
        labels_per_image[~anchors_per_image.get_field("visibility")] = -1

        # discard indices that are between thresholds
        inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
        labels_per_image[inds_to_discard] = -1

        # compute regression targets
        regression_targets_per_image = box_coder.encode(
            matched_targets.bbox, anchors_per_image.bbox
        )

        labels.append(labels_per_image)
        regression_targets.append(regression_targets_per_image)

    sampled_pos_inds, sampled_neg_inds = fg_bg_sampler(labels)
    sampled_pos_inds = torch.nonzero(
        torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
    sampled_neg_inds = torch.nonzero(
        torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

    sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    return labels, regression_targets, sampled_inds, sampled_pos_inds
