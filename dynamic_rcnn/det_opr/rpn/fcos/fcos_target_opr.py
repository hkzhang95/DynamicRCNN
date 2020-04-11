# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

INF = 100000000


def get_sample_region(
        gt, strides, num_points_per_level, gt_xs, gt_ys, radius=1):
    gt = gt[None].expand(gt_xs.shape[0], gt.shape[0], 4)
    center_x = (gt[..., 0] + gt[..., 2]) / 2
    center_y = (gt[..., 1] + gt[..., 3]) / 2
    center_gt = gt.new_zeros(gt.shape)
    # no gt
    if center_x[..., 0].sum() == 0:
        return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
    start = 0
    for level, num_points in enumerate(num_points_per_level):
        end = start + num_points
        stride = strides[level] * radius
        xmin = center_x[start:end] - stride
        ymin = center_y[start:end] - stride
        xmax = center_x[start:end] + stride
        ymax = center_y[start:end] + stride
        # limit sample region in gt
        center_gt[start:end, :, 0] = torch.where(
            xmin > gt[start:end, :, 0], xmin, gt[start:end, :, 0])
        center_gt[start:end, :, 1] = torch.where(
            ymin > gt[start:end, :, 1], ymin, gt[start:end, :, 1])
        center_gt[start:end, :, 2] = torch.where(
            xmax > gt[start:end, :, 2], gt[start:end, :, 2], xmax)
        center_gt[start:end, :, 3] = torch.where(
            ymax > gt[start:end, :, 3], gt[start:end, :, 3], ymax)
        start = end
    left = gt_xs[:, None] - center_gt[..., 0]
    right = center_gt[..., 2] - gt_xs[:, None]
    top = gt_ys[:, None] - center_gt[..., 1]
    bottom = center_gt[..., 3] - gt_ys[:, None]
    center_bbox = torch.stack((left, top, right, bottom), -1)
    inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
    return inside_gt_bbox_mask


def compute_targets_for_locations(
        locations, targets, object_sizes_of_interest, center_sample=None):
    labels = []
    reg_targets = []
    xs, ys = locations[:, 0], locations[:, 1]

    for im_i in range(len(targets)):
        targets_per_im = targets[im_i]
        assert targets_per_im.mode == "xyxy"
        bboxes = targets_per_im.bbox
        labels_per_im = targets_per_im.get_field("labels")
        area = targets_per_im.area()

        l = xs[:, None] - bboxes[:, 0][None]
        t = ys[:, None] - bboxes[:, 1][None]
        r = bboxes[:, 2][None] - xs[:, None]
        b = bboxes[:, 3][None] - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

        if center_sample is not None:
            fpn_strides = center_sample['fpn_strides']
            pos_radius = center_sample['pos_radius']
            num_points_per_level = center_sample['num_points_per_level']
            is_in_boxes = get_sample_region(
                bboxes, fpn_strides, num_points_per_level, xs, ys,
                radius=pos_radius)
        else:
            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

        max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
        # limit the regression range for each location
        is_cared_in_the_level = \
            (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
            (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

        locations_to_gt_area = area[None].repeat(len(locations), 1)
        locations_to_gt_area[is_in_boxes == 0] = INF
        locations_to_gt_area[is_cared_in_the_level == 0] = INF

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        locations_to_min_area, locations_to_gt_inds = \
            locations_to_gt_area.min(dim=1)

        reg_targets_per_im = reg_targets_per_im[
            range(len(locations)), locations_to_gt_inds]
        labels_per_im = labels_per_im[locations_to_gt_inds]
        labels_per_im[locations_to_min_area == INF] = 0

        labels.append(labels_per_im)
        reg_targets.append(reg_targets_per_im)

    return labels, reg_targets


def fcos_target_opr(locations, targets, center_sample=None):
    """
    Generate targets for computing fcos loss.

    Args:
        locations: (list[BoxList])
        targets: (list[BoxList])
        center_sample: (dict)
    """
    object_sizes_of_interest = [
        [-1, 64],
        [64, 128],
        [128, 256],
        [256, 512],
        [512, INF],
    ]
    expanded_object_sizes_of_interest = []
    for l, points_per_level in enumerate(locations):
        object_sizes_of_interest_per_level = \
            points_per_level.new_tensor(object_sizes_of_interest[l])
        expanded_object_sizes_of_interest.append(
            object_sizes_of_interest_per_level[None].expand(
                len(points_per_level), -1)
        )

    expanded_object_sizes_of_interest = torch.cat(
        expanded_object_sizes_of_interest, dim=0)
    num_points_per_level = [
        len(points_per_level) for points_per_level in locations]
    points_all_level = torch.cat(locations, dim=0)
    if center_sample is not None:
        center_sample['num_points_per_level'] = num_points_per_level
        labels, reg_targets = compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest,
            center_sample=center_sample)
    else:
        labels, reg_targets = compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest)

    for i in range(len(labels)):
        labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
        reg_targets[i] = torch.split(
            reg_targets[i], num_points_per_level, dim=0)

    labels_level_first = []
    reg_targets_level_first = []
    for level in range(len(locations)):
        labels_level_first.append(
            torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
        )
        reg_targets_level_first.append(
            torch.cat([reg_targets_per_im[level] for reg_targets_per_im in
                       reg_targets], dim=0)
        )
    return labels_level_first, reg_targets_level_first


def compute_centerness_targets(reg_targets):
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                  (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)
