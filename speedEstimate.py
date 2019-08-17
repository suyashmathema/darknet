import numpy as np
from utilities import *

frame = None
track_ids = None
detections = None
velocities = None
score = None
avg_velocities = None

s_x = 3.2/6.0
s_y = 1.5
y_min = 500.0
y_max = 1000.0
L0 = 0.2
f1 = 959.0
f2 = -182.2639
V0 = 40

# Specify H matrix to rectify videos of each location
H = [[L0, -L0*(f1/f2), 0.0], [0.0, 1.0, 0.0], [0.0, -(1/f2), 1.0]]


def compute_vel(box_vel, det, frame, s_x, s_y, y_min, y_max, H, last_frame, last_avg_vels):
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

    # tranlating the speed to pixel/second
    vi = instant_vel / t_delta * 30 * 9 / 4

    # Suppress noise
    if vi <= 3.0:
        vi = 0.0

    # Scale Recovery
    s1, s2 = 1.1, 0.9
    a, b = (s2 - s1) / (y_max - y_min), s1
    s = a * c[1] + b

    # Speed can be calculated using moving average which is more robust to noisy jitters in instant velocity
    # due to non-ideal detection and tracking
    records = len(last_avg_vels)
    v_estimate = ((last_avg_vels[-1] * records) + vi * s) / (records + 1)
    # v_estimate = (records + vi) / 2.0

    return v_estimate


def estimateSpeed(track_bbs_ids, frame_no):
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
        estimated_vels.append(compute_vel(box_vel[0], det[0], frame_no, s_x, s_y,
                                          y_min, y_max, H, last_frames, last_avg_vels))

    detections = np.append(detections, new_detections, 0)
    velocities = np.append(velocities, new_velocities, 0)
    score = np.append(score, new_score)
    track_ids = np.append(track_ids, new_track_ids)
    frame = np.append(frame, new_frame)
    avg_velocities = np.append(avg_velocities, estimated_vels)

    return estimated_vels
