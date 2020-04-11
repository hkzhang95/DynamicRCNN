import torch
from dynamic_rcnn.det_opr.matcher import Matcher
from dynamic_rcnn.det_opr.sampler import BalancedPositiveNegativeSampler
from dynamic_rcnn.datasets.structures.boxlist_ops import boxlist_iou


def proposal_target_opr(
        proposals, targets, box_coder, high_threshold, low_threshold,
        batch_size_per_image, positive_fraction, return_ious=False,
        return_sample_id=False, return_raw_proposals=False):
    """
    Generate proposal targets for computing loss.

    Args:
        proposals: (list[BoxList])
        targets: (list[BoxList])
        box_coder: (BoxCoder)
        high_threshold: (float)
        low_threshold: (float)
        batch_size_per_image: (int)
        positive_fraction: (float)
        return_ious: (bool)
    """

    matcher = Matcher(high_threshold, low_threshold,
                      allow_low_quality_matches=False)
    fg_bg_sampler = BalancedPositiveNegativeSampler(
        batch_size_per_image, positive_fraction)

    # prepare targets
    labels = []
    regression_targets = []
    ious = []
    for proposals_per_image, targets_per_image in zip(proposals, targets):
        # match targets to proposals
        match_quality_matrix = boxlist_iou(
            targets_per_image, proposals_per_image)
        matched_idxs = matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = targets_per_image.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_ious = match_quality_matrix.t()[
            range(proposals_per_image.bbox.shape[0]), matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)

        matched_idxs = matched_targets.get_field("matched_idxs")
        labels_per_image = matched_targets.get_field("labels")
        labels_per_image = labels_per_image.to(dtype=torch.int64)

        # Label background (below the low threshold)
        bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
        labels_per_image[bg_inds] = 0

        # Label ignore proposals (between low and high thresholds)
        ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
        labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

        # compute regression targets
        regression_targets_per_image = box_coder.encode(
            matched_targets.bbox, proposals_per_image.bbox
        )

        labels.append(labels_per_image)
        regression_targets.append(regression_targets_per_image)
        ious.append(matched_ious)

    sampled_pos_inds, sampled_neg_inds = fg_bg_sampler(labels)
    proposals = list(proposals)
    # add corresponding label and regression_targets information to the bounding boxes
    for labels_per_image, regression_targets_per_image, ious_per_image, \
        proposals_per_image in zip(labels, regression_targets, ious, proposals):
        proposals_per_image.add_field("labels", labels_per_image)
        proposals_per_image.add_field(
            "regression_targets", regression_targets_per_image
        )
        if return_ious:
            proposals_per_image.add_field("ious", ious_per_image)

    if return_sample_id:
        sample_id = []
    if return_raw_proposals:
        raw_proposals = proposals.copy()
    # distributed sampled proposals, that were obtained on all feature maps
    # concatenated via the fg_bg_sampler, into individual feature map levels
    for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
    ):
        img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
        proposals_per_image = proposals[img_idx][img_sampled_inds]
        proposals[img_idx] = proposals_per_image

        if return_sample_id:
            sample_id.append(img_sampled_inds)

    if return_sample_id:
        if return_raw_proposals:
            return proposals, sample_id, raw_proposals
        else:
            return proposals, sample_id
    else:
        if return_raw_proposals:
            return proposals, raw_proposals
        else:
            return proposals
