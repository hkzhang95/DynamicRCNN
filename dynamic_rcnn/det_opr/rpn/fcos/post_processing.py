import torch
from dynamic_rcnn.datasets.structures.bounding_box import BoxList
from dynamic_rcnn.datasets.structures.boxlist_ops import cat_boxlist, boxlist_nms, \
    remove_small_boxes


def post_processing_opr(
        fcos_locations, cls_logits, bbox_preds, centernesses, image_sizes,
        pre_nms_top_n, pre_nms_thresh, nms_thresh, box_min_size,
        fpn_post_nms_top_n, num_classes):
    """
    Compute the post-processed boxes and obtain the final results.

    Args:
        fcos_locations: (list[BoxList])
        cls_logits: (list[tensor])
        bbox_preds: (list[tensor])
        centernesses: (list[tensor])
        image_sizes: (list[tuple[int, int]])
        pre_nms_top_n: (int)
        pre_nms_thresh: (float)
        nms_thresh: (float)
        box_min_size: (int)
        fpn_post_nms_top_n: (int)
        num_classes: (int)
    """
    sampled_boxes = []
    temp_pre_nms_top_n = pre_nms_top_n

    for locations, box_cls, box_regression, centerness in zip(
            fcos_locations, cls_logits, bbox_preds, centernesses):

        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()

        candidate_inds = box_cls > pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=temp_pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, box_min_size)
            results.append(boxlist)
        sampled_boxes.append(results)

    boxlists = list(zip(*sampled_boxes))
    boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

    # select over all levels
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
