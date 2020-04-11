import torch
from dynamic_rcnn.datasets.structures.bounding_box import BoxList
from dynamic_rcnn.datasets.structures.boxlist_ops import cat_boxlist


# TODO: this should be implemented in RPN, but now a little different
def add_gt_proposals(proposals, targets):
    """
    Arguments:
        proposals: list[BoxList]
        targets: list[BoxList]
    """
    # Get the device we're operating on
    device = proposals[0].bbox.device

    gt_boxes = [target.copy_with_fields([]) for target in targets]

    # later cat of bbox requires all fields to be present for all bbox
    # so we need to add a dummy for objectness that's missing
    # check whether the proposal has the "objectness" field first
    if "objectness" in proposals[0].fields():
        for gt_box in gt_boxes:
            gt_box.add_field(
                "objectness", torch.ones(len(gt_box), device=device))

    proposals = [
        cat_boxlist((proposal, gt_box))
        for proposal, gt_box in zip(proposals, gt_boxes)
    ]

    return proposals


def add_box_regression(
        boxes, box_regression, box_coder, cls_agnostic_bbox_reg=False):
    if cls_agnostic_bbox_reg:
        box_regression = box_regression[:, -4:]

    boxes_per_image = [len(box) for box in boxes]
    concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)
    proposals = box_coder.decode(
        box_regression.view(sum(boxes_per_image), -1), concat_boxes)
    proposals = proposals.split(boxes_per_image, dim=0)

    result = []
    for img_id, proposal in enumerate(proposals):
        boxlist = BoxList(proposal, boxes[img_id].size, mode="xyxy")
        boxlist = boxlist.clip_to_image(remove_empty=False)
        result.append(boxlist)
    return result
