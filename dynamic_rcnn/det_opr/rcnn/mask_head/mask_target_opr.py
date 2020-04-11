import torch
from dynamic_rcnn.det_opr.matcher import Matcher
from dynamic_rcnn.datasets.structures.boxlist_ops import boxlist_iou


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )

    # FIXME: CPU computation bottleneck, this should be parallelized
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation.
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.get_mask_tensor()
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


def mask_target_opr(
        proposals, targets, high_threshold, low_threshold, discretization_size):
    """
    Generate proposal targets for computing loss.

    Args:
        proposals: (list[BoxList])
        targets: (list[BoxList])
        high_threshold: (float)
        low_threshold: (float)
        discretization_size: (int)
    """

    matcher = Matcher(high_threshold, low_threshold,
                      allow_low_quality_matches=False)

    # prepare targets
    labels = []
    masks = []
    for proposals_per_image, targets_per_image in zip(proposals, targets):
        # match targets to proposals
        match_quality_matrix = boxlist_iou(
            targets_per_image, proposals_per_image)
        matched_idxs = matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = targets_per_image.copy_with_fields(["labels", "masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)

        matched_idxs = matched_targets.get_field("matched_idxs")
        labels_per_image = matched_targets.get_field("labels")
        labels_per_image = labels_per_image.to(dtype=torch.int64)

        # this can probably be removed, but is left here for clarity
        # and completeness
        neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
        labels_per_image[neg_inds] = 0

        # mask scores are only computed on positive samples
        positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

        segmentation_masks = matched_targets.get_field("masks")
        segmentation_masks = segmentation_masks[positive_inds]

        positive_proposals = proposals_per_image[positive_inds]

        masks_per_image = project_masks_on_boxes(
            segmentation_masks, positive_proposals, discretization_size
        )

        labels.append(labels_per_image)
        masks.append(masks_per_image)

    return labels, masks
