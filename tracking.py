import pickle
import os
import scipy.io
import utilities
from sort import *
import argparse

MAX_DET_SIZE = 600
MIN_DET_SCORE = 0.3
NMS_THRESHOLD = 0.9

# Construct the dictionary to contain tracking results
tracking_dict = None
mot_tracker = None


def initialize_tracker():
    global tracking_dict, mot_tracker
    # Construct the dictionary to contain tracking results
    tracking_dict = {}
    tracking_dict['bbox'], tracking_dict['frame_num'], tracking_dict['track_id'] = [
    ], [], []
    tracking_dict['box_velocities'], tracking_dict['scores'] = [], []

    # reset the track ID of the tracker for new video
    KalmanBoxTracker.count = 0

    # establish the multi object tracker
    mot_tracker = Sort()


def filter_detections(detections):
    final_boxes = []
    # Filtering out only the detections for cars, trucks and buses
    for detection in detections:
        bb = detection[:5]
        lab = detection[5].decode("utf-8")
        bb.append(lab)
        if lab in ["bicycle", "car", "motorbike", "bus", "truck"]:
            final_boxes.append(bb)

    if len(final_boxes) == 0:
        return final_boxes

    # Convert the lists to numpy arrays
    final_boxes = np.asarray(final_boxes)

    # Imposing detection confidence and bounding box size to discard faulty boxes
    final_boxes = final_boxes[np.logical_and(
        final_boxes[:, 4].astype(np.float) > MIN_DET_SCORE, final_boxes[:, 2].astype(np.float) - final_boxes[:, 0].astype(np.float) < MAX_DET_SIZE)]

    # Apply NMS
    indices = utilities.non_max_suppression(
        final_boxes[:, :4], NMS_THRESHOLD, final_boxes[:, 4])

    if len(indices) == 0:
        return final_boxes

#     final_boxes = [final_boxes[i] for i in indices]
    final_boxes = final_boxes[indices]
    return final_boxes[:, :5].astype(np.float), final_boxes[:, 5]


def tracking(final_boxes, frame_no):
    global tracking_dict, mot_tracker
    # Update the tracker by feeding the current frame detected boxes
    track_bbs_ids = mot_tracker.update(np.array(final_boxes))

    if len(track_bbs_ids) == 0:
        return track_bbs_ids

    # Writing tracking results to the dictionary
    tracking_dict['bbox'].append(track_bbs_ids[:, :4])
    tracking_dict['box_velocities'].append(track_bbs_ids[:, -3:-1])
    tracking_dict['scores'].extend(track_bbs_ids[:, 4])
    tracking_dict['track_id'].extend(track_bbs_ids[:, 5])
    tracking_dict['frame_num'].extend([frame_no] * len(track_bbs_ids))
    return track_bbs_ids


def finalize_tracker():
    global tracking_dict
    # Stacking the tracking results for convenience
#     tracking_dict['bbox'] = np.vstack(tracking_dict['bbox'])
#     tracking_dict['box_velocities'] = np.vstack(
#         tracking_dict['box_velocities'])
