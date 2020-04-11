import torch
from dynamic_rcnn.utils.torch_utils import permute_and_flatten
from dynamic_rcnn.datasets.structures.bounding_box import BoxList
from dynamic_rcnn.datasets.structures.boxlist_ops import cat_boxlist, boxlist_nms, \
    remove_small_boxes


def post_processing_opr(
        retina_anchors, cls_logits, bbox_preds, box_coder, pre_nms_top_n,
        pre_nms_thresh, nms_thresh, box_min_size, fpn_post_nms_top_n,
        num_classes):
    """
    Compute the post-processed boxes and obtain the final results.

    Args:
        retina_anchors: (list[list[BoxList]])
        cls_logits: (list[tensor])
        bbox_preds: (list[tensor])
        box_coder: (BoxCoder)
        pre_nms_top_n: (int)
        pre_nms_thresh: (float)
        nms_thresh: (float)
        box_min_size: (int)
        fpn_post_nms_top_n: (int)
        num_classes: (int)
    """

    sampled_boxes = []
    num_levels = len(cls_logits)
    retina_anchors = list(zip(*retina_anchors))
    temp_pre_nms_top_n = pre_nms_top_n
    for anchors, box_cls, box_regression in zip(
            retina_anchors, cls_logits, bbox_preds):
        device = box_cls.device
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A

        # put in the same format as anchors
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)

        num_anchors = A * H * W

        candidate_inds = box_cls > pre_nms_thresh

        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=temp_pre_nms_top_n)

        results = []
        for per_box_cls, per_box_regression, per_pre_nms_top_n, \
            per_candidate_inds, per_anchors in zip(
            box_cls,
            box_regression,
            pre_nms_top_n,
            candidate_inds,
            anchors):
            # Sort and select TopN
            per_box_cls = per_box_cls[per_candidate_inds]

            per_box_cls, top_k_indices = \
                per_box_cls.topk(per_pre_nms_top_n, sorted=False)

            per_candidate_nonzeros = \
                per_candidate_inds.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]
            per_class += 1

            detections = box_coder.decode(
                per_box_regression[per_box_loc, :].view(-1, 4),
                per_anchors.bbox[per_box_loc, :].view(-1, 4)
            )

            boxlist = BoxList(detections, per_anchors.size, mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, box_min_size)
            results.append(boxlist)
        sampled_boxes.append(results)

    boxlists = list(zip(*sampled_boxes))
    boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

    # select over all levels
    if num_levels > 1:
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            # skip the background
            for j in range(1, num_classes):
                inds = (labels == j).nonzero().view(-1)

                scores_j = scores[inds]
                boxes_j = boxes[inds, :].view(-1, 4)
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, nms_thresh,
                    score_field="scores"
                )
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j,
                                         dtype=torch.int64,
                                         device=scores.device)
                )
                result.append(boxlist_for_class)

            result = cat_boxlist(result)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results
    return boxlists
