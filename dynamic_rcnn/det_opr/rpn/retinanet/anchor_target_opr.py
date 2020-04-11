# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from dynamic_rcnn.det_opr.matcher import Matcher
from dynamic_rcnn.datasets.structures.boxlist_ops import boxlist_iou


def anchor_target_opr(
        anchors, targets, box_coder, high_threshold, low_threshold,
        allow_low_quality_matches=True):
    """
    Generate anchor targets for computing retinanet loss.

    Args:
        anchors: (list[BoxList])
        targets: (list[BoxList])
        box_coder: (BoxCoder)
        high_threshold: (float)
        low_threshold: (float)
    """
    matcher = Matcher(high_threshold, low_threshold,
                      allow_low_quality_matches=allow_low_quality_matches)

    # prepare targets
    labels = []
    regression_targets = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        # match targets to anchors
        match_quality_matrix = boxlist_iou(targets_per_image, anchors_per_image)
        matched_idxs = matcher(match_quality_matrix)
        targets_per_image = targets_per_image.copy_with_fields(['labels'])
        matched_targets = targets_per_image[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)

        matched_idxs = matched_targets.get_field("matched_idxs")
        # generate rpn labels
        labels_per_image = matched_targets.get_field("labels")
        labels_per_image = labels_per_image.to(dtype=torch.float32)

        # Background (negative examples)
        bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
        labels_per_image[bg_indices] = 0

        # discard indices that are between thresholds
        inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
        labels_per_image[inds_to_discard] = -1

        # compute regression targets
        regression_targets_per_image = box_coder.encode(
            matched_targets.bbox, anchors_per_image.bbox
        )

        labels.append(labels_per_image)
        regression_targets.append(regression_targets_per_image)

    return labels, regression_targets
