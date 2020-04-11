# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import errno
import os
import cv2


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def link_file(src, target):
    """symbol link the source directories to target."""
    if os.path.isdir(target) or os.path.isfile(target):
        os.remove(target)
    os.system('ln -s {} {}'.format(src, target))


def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy


def draw_box(image, box, label, color=(0, 0, 255), score=None, linewidth=2):
    """Draw a bounding box with label on the image."""
    if score is not None:
        text = "{}: {:.4f}".format(label, score)
    else:
        text = str(label)

    cv2.rectangle(image, (int(box[0]), int(box[1])),
                  (int(box[2]), int(box[3])), color, linewidth)
    cx = box[0] + (box[2] - box[0]) / 2 - 5
    cy = box[1] + 12
    cv2.putText(image, text, (int(cx), int(cy)),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))