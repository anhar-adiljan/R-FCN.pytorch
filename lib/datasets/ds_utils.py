# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Adilijiang (Adil) Ainihaer
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import numpy as np

_box_formats_ = ['xywh', 'xyxy']

def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)

def convert_box_format(boxes, src_fmt, dst_fmt):
    # validate format inputs
    _validate_box_format(src_fmt)
    _validate_box_format(dst_fmt)
    assert src_fmt != dst_fmt

    if src_fmt == 'xywh' and dst_fmt == 'xyxy':
        return _xywh_to_xyxy(boxes)

    if src_fmt == 'xyxy' and dst_fmt == 'xywh':
        return _xyxy_to_xywh(boxes)

def _validate_box_format(fmt):
    assert fmt in _box_formats_

def _xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def _xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

def validate_boxes(boxes, width=0, height=0, fmt='xyxy'):
    """Check that a set of boxes are valid."""
    _validate_box_format(fmt)

    if fmt == 'xyxy':
        _validate_xyxy_boxes(boxes, width, height)

    if fmt == 'xywh':
        _validate_xywh_boxes(boxes, width, height)

def _validate_xywh_boxes(boxes, width=0, height=0):
    boxes = _xywh_to_xyxy(boxes)
    _validate_xyxy_boxes(boxes)

def _validate_xyxy_boxes(boxes, width=0, height=0):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # check if coordinates are in bounds
    _validate_coordinates(x1, lower_bound=0, upper_bound=width)
    _validate_coordinates(y1, lower_bound=0, upper_bound=height)
    _validate_coordinates(x2, lower_bound=x1, upper_bound=width)
    _validate_coordinates(y2, lower_bound=y1, upper_bound=height)

def _validate_coordinates(x, lower_bound, upper_bound):
    assert (x >= lower_bound).all() and (x < upper_bound).all()

def filter_small_boxes(boxes, min_size, fmt='xyxy'):
    _validate_box_format(fmt)

    w = boxes[:, 2] - boxes[:, 0] if fmt == 'xyxy' else boxes[:, 2]
    h = boxes[:, 3] - boxes[:, 1] if fmt == 'xyxy' else boxes[:, 3]
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep
