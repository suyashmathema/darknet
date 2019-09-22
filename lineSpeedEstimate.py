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

csv_data = np.empty([1, 7])
inter_data = {}
dist = 6
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
    global csv_data, inter_data
    new_detections = track_bbs_ids[:, :4]
    new_score = track_bbs_ids[:, 4]
    new_track_ids = track_bbs_ids[:, 5]

    prev_frame = csv_data[csv_data[:, 0] == (frame_no - 1), :]

    for tid in new_track_ids:
        curr_det = new_detections[new_track_ids == tid][0]
        det = convertToCenterCoord(curr_det)
        prev_coord = prev_frame[prev_frame[:, 1] == tid]

        data = [frame_no, tid, det[0], det[1], det[2],
                         det[3], new_score[new_track_ids == tid][0], False, False]

        if intersect((det[0], det[1]), (prev_coord[2], prev_coord[3]), line1[0], line1[1]):
            inter_data[tid]["line1"] = frame_no
            data[7] = 1

        if intersect((det[0], det[1]), (prev_coord[2], prev_coord[3]), line2[0], line2[1]):
            inter_data[tid]["line2"] = frame_no
            inter_data[tid]["speed"] = (
                frame_no - inter_data[tid]["line1"]) / fps * 3.6
            data[7] = 2
            data[8] = True

        csv_data.append()

    return inter_data

def getSpeedLines():
    return (line1, line2)

def getCsvData():
    return csv_data
