import numpy as np
from utilities import *

frame = None
track_ids = None
detections = None
velocities = None
score = None
avg_velocities = None

s_x = 16/223
s_y = 3.9
y_min = 0
y_max = 1500.0
L0 = 1
f1 = 767.5
f2 = -143.5
V0 = 40

# Specify H matrix to rectify videos of each location
H = [[L0, -L0*(f1/f2), 0.0], [0.0, 1.0, 0.0], [0.0, -(1/f2), 1.0]]


parameters = {'GOPR1496-Car-60': {"s_x": 16/115, "s_y": 3.9, "y_min": 0.0, "y_max": 1500.0, "L0": 1, "f1": 517.5, "f2": -85.5, "V0": 40, "s1": 1.5/3, "s2": 1.5/2},
              'GOPR1495-Car-50': {"s_x": 16/165, "s_y": 3.9, "y_min": 25.0, "y_max": 475.0, "L0": 1, "f1": 477.5, "f2": -127.5, "V0": 40, "s1": 1.5/4, "s2": 1.5/2},
              'GOPR1494-Car-50': {"s_x": 16/240, "s_y": 3.9, "y_min": 30.0, "y_max": 680.0, "L0": 1, "f1": 782.5, "f2": -163.5, "V0": 40, "s1": 1.5/7, "s2": 1.5/5},
              'GOPR1493-Car-50': {"s_x": 16/267, "s_y": 3.9, "y_min": 30.0, "y_max": 680.0, "L0": 1, "f1": 627.5, "f2": -185.5, "V0": 40, "s1": 1.5/7, "s2": 1.5/5},
              'GOPR1492-Car-40': {"s_x": 16/217, "s_y": 3.9, "y_min": 30.0, "y_max": 680.0, "L0": 1, "f1": 757.5, "f2": -146.5, "V0": 40, "s1": 1.5/5, "s2": 1.5/3},
              'GOPR1491-Car-30': {"s_x": 16/262, "s_y": 3.9, "y_min": 30.0, "y_max": 680.0, "L0": 1, "f1": 607.5, "f2": -180.5, "V0": 40, "s1": 1.5/7, "s2": 1.5/5},
              'GOPR1489-Bike-60': {"s_x": 16/133, "s_y": 3.9, "y_min": 25.0, "y_max": 425.0, "L0": 1, "f1": 472.5, "f2": -94.5, "V0": 40, "s1": 1.5/3, "s2": 1.5/2},
              'GOPR1488-Bike-50': {"s_x": 16/199, "s_y": 3.9, "y_min": 30.0, "y_max": 530.0, "L0": 1, "f1": 712.5, "f2": -126.5, "V0": 40, "s1": 1.5/5, "s2": 1.5/3},
              'GOPR1487-Bike-50': {"s_x": 16/294, "s_y": 3.9, "y_min": 30.0, "y_max": 530.0, "L0": 1, "f1": 597.5, "f2": -196.5, "V0": 40, "s1": 1.5/8, "s2": 1.5/6},
              'GOPR1486-Bike-40': {"s_x": 16/189, "s_y": 3.9, "y_min": 30.0, "y_max": 680.0, "L0": 1, "f1": 787.5, "f2": -127.5, "V0": 40, "s1": 1.5/5, "s2": 1.5/4},
              'GOPR1485-Bike-30': {"s_x": 16/178, "s_y": 3.9, "y_min": 30.0, "y_max": 530.0, "L0": 1, "f1": 587.5, "f2": -114.5, "V0": 40, "s1": 1.5/5, "s2": 1.5/3}
              }


def init_speed_param(inputName):
    global s_x, s_y, y_min, y_max, L0, f1, f2, V0, s1, s2, H
    param = parameters[inputName]  
    s_x = param["s_x"]
    s_y = param["s_y"]
    y_min = param["y_min"]
    y_max = param["y_max"]
    L0 = param["L0"]
    f1 = param["f1"]
    f2 = param["f2"]
    V0 = param["V0"]
    s1 = param["s1"]
    s2 = param["s2"]
    H = [[L0, -L0*(f1/f2), 0.0], [0.0, 1.0, 0.0], [0.0, -(1/f2), 1.0]]


def compute_vel(box_vel, det, frame, s_x, s_y, s1, s2, y_min, y_max, H, last_frame, last_avg_vels, fps):
    """
    det: (x_1, y_1, x_2, y_2)
    """
    if len(last_frame) == 0:
        return V0

    v = box_vel

    c = np.array([(det[0] + det[2]) / 2.0, det[3] - y_min])

    # Transform the Velocities using the computed camera to real world equations in the paper
    v_x_trans = (((H[0][0] * v[0] + H[0][1] * v[1]) * (H[2][0] * c[0] + H[2][1] * c[1] + H[2][2]) -
                  (H[0][0] * c[0] + H[0][1] * c[1] + H[0][2]) * (H[2][0] * v[0] + H[2][1] * v[1])) /
                 ((H[2][0] * c[0] + H[2][1] * c[1] + H[2][2]) ** 2))
    v_y_trans = (((H[1][0] * v[0] + H[1][1] * v[1]) * (H[2][0] * c[0] + H[2][1] * c[1] + H[2][2]) -
                  (H[1][0] * c[0] + H[1][1] * c[1] + H[1][2]) * (H[2][0] * v[0] + H[2][1] * v[1])) /
                 ((H[2][0] * c[0] + H[2][1] * c[1] + H[2][2]) ** 2))

    # Velocity calculation
    instant_vel = np.sqrt(
        sum([(v_x_trans * s_x) ** 2, (v_y_trans * s_y) ** 2]))
    t_delta = frame - last_frame[-1]

    # translating the speed to pixel/second
    vi = instant_vel / t_delta * fps * 3.6

    # Suppress noise
    if vi <= 3.0:
        vi = 0.0

    # Scale Recovery
    # s1, s2 = 1.5/6, 1.5/4
    a, b = (s2 - s1) / (y_max - y_min), s1
    s = a * c[1] + b

    # Speed can be calculated using moving average which is more robust to noisy jitters in instant velocity
    # due to non-ideal detection and tracking
    records = len(last_avg_vels)
    v_estimate = ((last_avg_vels[-1] * records) + vi * s) / (records + 1)
    # v_estimate = (records + vi) / 2.0

    return v_estimate


def estimateSpeed(track_bbs_ids, frame_no, fps):
    global frame, track_ids, detections, velocities, score, avg_velocities
    new_detections = track_bbs_ids[:, :4]
    new_velocities = track_bbs_ids[:, -3:-1]
    new_score = track_bbs_ids[:, 4]
    new_track_ids = track_bbs_ids[:, 5]
    new_frame = [frame_no] * len(track_bbs_ids)

    if (detections is None):
        detections = np.matrix(new_detections)
        velocities = np.matrix(new_velocities)
        score = np.array(new_score)
        track_ids = np.array(new_track_ids)
        frame = np.array(new_frame)
        avg_velocities = np.array([V0] * len(track_bbs_ids))
        print('no previous estimates', avg_velocities)
        return avg_velocities

    # # filter out bounding boxes that is not in the cropped image
    # detections_filtered, track_ids_filtered, velocities_filtered = [], [], []

    # # Only Consider the detection and tracks in the region of interest
    # for i in range(len(new_detections)):
    #     det = new_detections[i]
    #     if det[1] > y_min:
    #         detections_filtered.append(det)
    #         velocities_filtered.append(velocities[i])
    #         track_ids_filtered.append(track_ids[i])

    # # Convert to numpy arrays
    # detections_filtered = np.matrix(detections_filtered)
    # velocities_filtered = np.matrix(velocities_filtered)
    # track_ids_filtered = np.array(track_ids_filtered)

    estimated_vels = []
    for tr in new_track_ids:
        det = new_detections[new_track_ids == tr]
        box_vel = new_velocities[new_track_ids == tr]
        last_frames = frame[track_ids == tr]
        last_avg_vels = avg_velocities[track_ids == tr]
        # measure the velocity
        estimated_vels.append(compute_vel(box_vel[0], det[0], frame_no, s_x, s_y, s1, s2,
                                          y_min, y_max, H, last_frames, last_avg_vels, fps))

    detections = np.append(detections, new_detections, 0)
    velocities = np.append(velocities, new_velocities, 0)
    score = np.append(score, new_score)
    track_ids = np.append(track_ids, new_track_ids)
    frame = np.append(frame, new_frame)
    avg_velocities = np.append(avg_velocities, estimated_vels)

    return estimated_vels
