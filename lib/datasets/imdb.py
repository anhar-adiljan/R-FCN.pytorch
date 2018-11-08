# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# Modified by Adilijiang (Adil) Ainihaer
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import PIL
from model.utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse as sparse
from model.utils.config import cfg
import pdb

ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')

class imdb(object):
    """Image database."""

    def __init__(self, name, classes=None):
        self._name = name
        self._num_classes = 0
        if not classes:
            self._classes = []
        else:
            self._classes = classes
        self._image_index = []
        self._obj_proposer = 'gt'
        self._roidb = None
        self._roidb_handler = self.default_roidb
        # Use this dict for storing dataset specific config options
        self.config = {}

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')
        self.roidb_handler = method

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
        return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def image_id_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def _get_width_by_index(self, image_idx):
        """Returns width of the image associated with the given index."""
        return PIL.Image.open(self.image_path_at(image_idx)).size[0]

    def _get_widths(self):
        """Returns widths of all images in the database."""
        return [PIL.Image.open(self.image_path_at(i)).size[0]
                for i in range(self.num_images)]

    def _get_box_coords(self, boxes):
        """Returns copies of the coordinates of the given boxes."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return x1.copy(), y1.copy(), x2.copy(), y2.copy()

    def _compute_flipped_coords(self, width, old_x1, old_x2):
        """Returns the new coordinates of boxes after horizontal flip."""
        new_x1 = width - old_x2 - 1
        new_x2 = width - old_x1 - 1
        return new_x1, new_x2

    def _flip_image_boxes(self, image_idx):
        """Horizontally flip boxes associated with image at given index."""
        width = self._get_width_by_index(image_idx)
        boxes = self.roidb[image_idx]['boxes'].copy()
        x1, _, x2, _ = self._get_box_coords(boxes)
        boxes[:, 0], boxes[:, 2] = self._compute_flipped_coords(width, x1, x2)
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        return boxes

    def _create_flipped_entry(self, image_idx):
        """Returns a dictionary representing an RoI flipped entry."""
        boxes = self._flip_image_boxes(image_idx)
        gt_overlaps = self.roidb[image_idx]['gt_overlaps']
        gt_classes = self.roidb[image_idx]['gt_classes']
        flipped = True
        return {'boxes': boxes, 'gt_overlaps': gt_overlaps,
                'gt_classes': gt_classes, 'flipped': True}

    def _add_roidb_flipped_entry(self, image_idx):
        """Adds a flipped entry to the RoI database."""
        entry = self._create_flipped_entry(image_idx)
        self.roidb.append(entry)

    def append_flipped_images(self):
        """Adds flipped images to the image database."""
        num_images = self.num_images
        for i in range(num_images):
            self._add_roidb_flipped_entry(i)
        # Expand image indices by factor of 2 after adding all flipped images.
        self._image_index = self._image_index * 2

    def evaluate_recall(self, candidate_boxes=None, thresholds=None,
                        area='all', limit=None):
        """Evaluate detection proposal recall metrics.

        Returns:
            results: dictionary of results with keys
                'ar': average recall
                'recalls': vector recalls at each IoU overlap threshold
                'thresholds': vector of IoU overlap thresholds
                'gt_overlaps': vector of all ground-truth overlaps
        """
        # Record max overlap value for each gt box
        # Return vector of overlap values
        areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3,
                 '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}

        area_ranges = [[0 ** 2, 1e5 ** 2],  # all
                       [0 ** 2, 32 ** 2],  # small
                       [32 ** 2, 96 ** 2],  # medium
                       [96 ** 2, 1e5 ** 2],  # large
                       [96 ** 2, 128 ** 2],  # 96-128
                       [128 ** 2, 256 ** 2],  # 128-256
                       [256 ** 2, 512 ** 2],  # 256-512
                       [512 ** 2, 1e5 ** 2],  # 512-inf
                       ]

        assert area in areas, 'unknown area range: {}'.format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = np.zeros(0)
        num_pos = 0
        for i in range(self.num_images):
            # Checking for max_overlaps == 1 avoids including crowd annotations
            # (...pretty hacking :/)
            max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)
            gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
                               (max_gt_overlaps == 1))[0]
            gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
            gt_areas = self.roidb[i]['seg_areas'][gt_inds]
            valid_gt_inds = np.where((gt_areas >= area_range[0]) &
                                     (gt_areas <= area_range[1]))[0]
            gt_boxes = gt_boxes[valid_gt_inds, :]
            num_pos += len(valid_gt_inds)

            if candidate_boxes is None:
                # If candidate_boxes is not supplied, the default is to use the
                # non-ground-truth boxes from this roidb
                non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
                boxes = self.roidb[i]['boxes'][non_gt_inds, :]
            else:
                boxes = candidate_boxes[i]
            if boxes.shape[0] == 0:
                continue
            if limit is not None and boxes.shape[0] > limit:
                boxes = boxes[:limit, :]

            overlaps = bbox_overlaps(boxes.astype(np.float),
                                     gt_boxes.astype(np.float))

            _gt_overlaps = np.zeros((gt_boxes.shape[0]))

            for j in range(gt_boxes.shape[0]):
                # find which proposal box maximally covers each gt box
                argmax_overlaps = overlaps.argmax(axis=0)
                # and get the iou amount of coverage for each gt box
                max_overlaps = overlaps.max(axis=0)
                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ind = max_overlaps.argmax()
                gt_ovr = max_overlaps.max()
                assert (gt_ovr >= 0)
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert (_gt_overlaps[j] == gt_ovr)
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1
            # append recorded iou coverage level
            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

        gt_overlaps = np.sort(gt_overlaps)
        if thresholds is None:
            step = 0.05
            thresholds = np.arange(0.5, 0.95 + 1e-5, step)

        recalls = np.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
                'gt_overlaps': gt_overlaps}

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == self.num_images, \
            'Number of boxes must match number of ground-truth images'
        roidb = []
        for i in range(self.num_images):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

        if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
            gt_boxes = gt_roidb[i]['boxes']
            gt_classes = gt_roidb[i]['gt_classes']
            gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                        gt_boxes.astype(np.float))
            argmaxes = gt_overlaps.argmax(axis=1)
            maxes = gt_overlaps.max(axis=1)
            I = np.where(maxes > 0)[0]
            overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

        overlaps = sparse.csr_matrix(overlaps)
        roidb.append({
            'boxes': boxes,
            'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': np.zeros((num_boxes,), dtype=np.float32),
        })

        return roidb

    @staticmethod
    def merge_roidb_entry(a, b):
        """Merges two roidb entries into one and returns the merged entry."""
        a['boxes'] = np.vstack((a['boxes'], b['boxes']))
        a['gt_classes'] = np.hstack((a['gt_classes'], b['gt_classes']))
        a['gt_overlaps'] = sparse.vstack([['gt_overlaps'], b['gt_overlaps']])
        a['seg_areas'] = np.hstack((a['seg_areas'], b['seg_areas']))
        return a

    @staticmethod
    def merge_roidbs(a, b):
        """Merges two roidbs into one and returns the merged db."""
        assert len(a) == len(b)
        # Iterates through the two dbs and merged each pair of entries.
        for i in range(len(a)):
            a[i] = imdb.merge_roidb_entry(a[i], b[i])
        return a

    def competition_mode(self, on):
        """Turn competition mode on or off."""
        pass
