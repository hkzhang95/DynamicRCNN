import torch
import torch.nn.functional as F

from dynamic_rcnn.datasets.structures.bounding_box import BoxList
from dynamic_rcnn.datasets.structures.boxlist_ops import boxlist_nms, cat_boxlist


def filter_results(
        boxlist, num_classes, score_thresh, nms_thresh, detections_per_img):
    # unwrap the boxlist to avoid additional overhead.
    # if we had multi-class NMS, we could perform this directly on the boxlist
    boxes = boxlist.bbox.reshape(-1, num_classes * 4)
    scores = boxlist.get_field("scores").reshape(-1, num_classes)

    device = scores.device
    result = []
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    inds_all = scores > score_thresh
    for j in range(1, num_classes):
        inds = inds_all[:, j].nonzero().squeeze(1)
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4: (j + 1) * 4]
        boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
        boxlist_for_class.add_field("scores", scores_j)
        boxlist_for_class = boxlist_nms(boxlist_for_class, nms_thresh)
        num_labels = len(boxlist_for_class)
        boxlist_for_class.add_field(
            "labels",
            torch.full((num_labels,), j, dtype=torch.int64, device=device)
        )
        result.append(boxlist_for_class)

    result = cat_boxlist(result)
    number_of_detections = len(result)

    # Limit to max_per_image detections **over all classes**
    if number_of_detections > detections_per_img > 0:
        cls_scores = result.get_field("scores")
        image_thresh, _ = torch.kthvalue(
            cls_scores.cpu(), number_of_detections - detections_per_img + 1
        )
        keep = cls_scores >= image_thresh.item()
        keep = torch.nonzero(keep).squeeze(1)
        result = result[keep]
    return result


# TODO: merge into test
def post_processing_opr(boxes, logits, offsets, box_coder, score_thresh=0.05,
                        nms_thresh=0.5, detections_per_img=100,
                        cls_agnostic_bbox_reg=False, bbox_aug_enabled=False):
    """
    Compute the post-processed boxes and obtain the final results.

    Args:
        boxes: (list[BoxList])
        logits: (tensor)
        offsets: (tensor)
        box_coder: (BoxCoder)
        score_thresh: (float)
        nms_thresh: (float)
        detections_per_img: (int)
        cls_agnostic_bbox_reg: (bool)

    Returns:
        results: (list[BoxList])
    """

    class_prob = F.softmax(logits, -1)
    num_classes = class_prob.shape[1]

    image_shapes = [box.size for box in boxes]
    boxes_per_image = [len(box) for box in boxes]
    concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

    if cls_agnostic_bbox_reg:
        offsets = offsets[:, -4:]
    proposals = box_coder.decode(
        offsets.view(sum(boxes_per_image), -1), concat_boxes)
    if cls_agnostic_bbox_reg:
        proposals = proposals.repeat(1, num_classes)

    proposals = proposals.split(boxes_per_image, dim=0)
    class_prob = class_prob.split(boxes_per_image, dim=0)

    results = []
    for prob, proposal, image_shape in zip(class_prob, proposals, image_shapes):
        # prepare boxlist
        proposal = proposal.reshape(-1, 4)
        prob = prob.reshape(-1)
        boxlist = BoxList(proposal, image_shape, mode="xyxy")
        boxlist.add_field("scores", prob)

        # clip tp image
        boxlist = boxlist.clip_to_image(remove_empty=False)

        # filter results
        if not bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
            boxlist = filter_results(boxlist, num_classes, score_thresh,
                                     nms_thresh, detections_per_img)
        results.append(boxlist)
    return results
