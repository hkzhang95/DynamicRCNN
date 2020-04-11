import torch
from dynamic_rcnn.utils.torch_utils import permute_and_flatten
from dynamic_rcnn.datasets.structures.bounding_box import BoxList
from dynamic_rcnn.datasets.structures.boxlist_ops import cat_boxlist, boxlist_nms, \
    remove_small_boxes


def proposal_opr(
        rpn_anchors, rpn_cls_logits, rpn_bbox_preds, box_coder, pre_nms_top_n,
        post_nms_top_n, nms_thresh, box_min_size, fpn_post_nms_top_n,
        fpn_post_nms_per_batch=True, is_train=False, targets=None,
        proposal_with_gt=True):
    """
    Generate proposals for RCNN.

    Args:
        rpn_anchors: (list[list[BoxList]])
        rpn_cls_logits: (list[tensor])
        rpn_bbox_preds: (list[tensor])
        box_coder: (BoxCoder)
        pre_nms_top_n: (int)
        post_nms_top_n: (int)
        nms_thresh: (float)
        box_min_size: (int)
        fpn_post_nms_top_n: (int)
        fpn_post_nms_per_batch: (bool)
        is_train: (bool)
        targets: (list[BoxList])
        proposal_with_gt: (bool)
    """

    sampled_boxes = []
    num_levels = len(rpn_cls_logits)
    rpn_anchors = list(zip(*rpn_anchors))
    for anchors, objectness, box_regression in zip(
            rpn_anchors, rpn_cls_logits, rpn_bbox_preds):
        device = objectness.device
        N, A, H, W = objectness.shape

        # put in the same format as anchors
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)
        objectness = objectness.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)

        num_anchors = A * H * W

        pre_nms_top_n = min(pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(
            pre_nms_top_n, dim=1, sorted=True)

        batch_idx = torch.arange(N, device=device)[:, None]
        box_regression = box_regression[batch_idx, topk_idx]

        image_shapes = [box.size for box in anchors]
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]

        proposals = box_coder.decode(
            box_regression.view(-1, 4), concat_anchors.view(-1, 4)
        )

        proposals = proposals.view(N, -1, 4)

        result = []
        for proposal, score, im_shape in zip(proposals, objectness,
                                             image_shapes):
            boxlist = BoxList(proposal, im_shape, mode="xyxy")
            boxlist.add_field("objectness", score)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, box_min_size)
            boxlist = boxlist_nms(
                boxlist,
                nms_thresh,
                max_proposals=post_nms_top_n,
                score_field="objectness",
            )
            result.append(boxlist)
        sampled_boxes.append(result)

    boxlists = list(zip(*sampled_boxes))
    boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

    # select over all levels
    if num_levels > 1:
        num_images = len(boxlists)
        if is_train and fpn_post_nms_per_batch:
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0,
                                        sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                post_nms_top_n = min(fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                boxlists[i] = boxlists[i][inds_sorted]

    # append ground-truth bboxes to proposals
    if is_train and targets is not None and proposal_with_gt:
        # Get the device we're operating on
        device = boxlists[0].bbox.device

        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness",
                             torch.ones(len(gt_box), device=device))

        boxlists = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(boxlists, gt_boxes)
        ]

    return boxlists
