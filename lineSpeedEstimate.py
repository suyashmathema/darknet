import numpy as np
from utilities import *

# Return true if line segments AB and CD intersect


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


parameters = {'GOPR1496-Car-60': {"dist": 10, "line1": [(491, 217), (763, 203)], "line2": [(479, 392), (896, 370)]},
              'GOPR1495-Car-50': {"dist": 10, "line1": [(498, 185), (760, 166)], "line2": [(514, 362), (907, 326)]},
              'GOPR1494-Car-50': {"dist": 10, "line1": [(747, 237), (1145, 233)], "line2": [(729, 474), (1336, 461)]},
              'GOPR1493-Car-50': {"dist": 10, "line1": [(654, 224), (1039, 215)], "line2": [(671, 459), (1239, 432)]},
              'GOPR1492-Car-40': {"dist": 10, "line1": [(720, 255), (1107, 243)], "line2": [(701, 500), (1282, 466)]},
              'GOPR1491-Car-30': {"dist": 10, "line1": [(652, 233), (1016, 213)], "line2": [(672, 473), (1230, 417)]}
              }

frame = None
track_ids = None
detections = None
score = None
inter_data = None
dist = 10
line1 = []
line2 = []


def init_speed_param(inputName):
    global dist, line1, line2
    param = parameters[inputName]
    dist = param["dist"]
    line1 = param["line1"]
    line2 = param["line2"]


def convertToCenterCoord(detection):
    xmin = detection[0]
    ymin = detection[1]
    xmax = detection[2]
    ymax = detection[3]
    x = (xmin+xmax)/2
    y = (ymin+ymax)/2
    w = xmax - xmin
    h = ymax - ymin
    return [x, y, w, h]


def estimateSpeed(track_bbs_ids, frame_no, fps):
    global frame, track_ids, detections, score, inter_data
    new_detections = track_bbs_ids[:, :4]
    new_score = track_bbs_ids[:, 4]
    new_track_ids = track_bbs_ids[:, 5]
    new_frame = [frame_no] * len(track_bbs_ids)
    new_inter_data = np.array(
        [[-1.0, -1.0, -1.0, frame_no]] * len(track_bbs_ids))

    if (detections is None):
        detections = np.matrix(new_detections)
        score = np.array(new_score)
        track_ids = np.array(new_track_ids)
        frame = np.array(new_frame)
        inter_data = np.matrix(new_inter_data)
        return new_inter_data

    prev_dets = detections[frame == frame_no - 1]
    prev_tids = track_ids[frame == frame_no - 1]

    for i, tid in enumerate(new_track_ids):
        curr_det = new_detections[new_track_ids == tid]
        prev_det = np.asarray(prev_dets[prev_tids == tid])

        if len(prev_det) > 0:
            curr_coord = convertToCenterCoord(curr_det[0])
            prev_coord = convertToCenterCoord(prev_det[0])

            prev_inter_id = np.asarray(inter_data[track_ids == tid])
            new_inter_data[i] = np.amax(prev_inter_id, axis=0)
            new_inter_data[i][3] = frame_no

            if intersect((curr_coord[0], curr_coord[1]), (prev_coord[0], prev_coord[1]), line1[0], line1[1]):
                new_inter_data[i][0] = frame_no

            if intersect((curr_coord[0], curr_coord[1]), (prev_coord[0], prev_coord[1]), line2[0], line2[1]):
                new_inter_data[i][1] = frame_no
                line1Frame = prev_inter_id[prev_inter_id[:, 0] > 0][0][0]
                new_inter_data[i][2] = 10.0 / \
                    (frame_no - line1Frame) * fps * 3.6
                print("Calculating speed", tid, line1Frame,
                      frame_no, new_inter_data[i][2])

    detections = np.append(detections, new_detections, 0)
    score = np.append(score, new_score)
    track_ids = np.append(track_ids, new_track_ids)
    frame = np.append(frame, new_frame)
    inter_data = np.append(inter_data, new_inter_data, 0)
    return new_inter_data


def getSpeedLines():
    return (line1, line2)


def getCsvData():
    csv_data = []
    for i, tid in enumerate(track_ids):
        csv_data.append({
            "frame": frame[i],
            "tid": tid,
            "speed": inter_data[i, 2],
            "line1": inter_data[i, 0],
            "line2": inter_data[i, 1],
            "xmin": detections[i, 0],
            "ymin": detections[i, 1],
            "xmax": detections[i, 2],
            "ymax": detections[i, 3],
            "score": score[i],
        })
    return csv_data
