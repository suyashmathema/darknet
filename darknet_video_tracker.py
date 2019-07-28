from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

from sort import *
tracker = Sort()

from nms import *

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

def convertToDetectionFormat(detections, size):
    W_scale = size[0] /416
    H_scale = size[1] /416
    dets_tracker = []
    dets_nms = []
    dets_confidence = []
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        # scale the bounding box coordinates back relative to
        # the size of the image, keeping in mind that YOLO
        # actually returns the center (x, y)-coordinates of
        # the bounding box followed by the boxes' width and
        # height
        box = [x, y, w, h] * np.array([W_scale, H_scale, W_scale, H_scale])
        # (centerX, centerY, width, height) = box.astype("int")
        xmin, ymin, xmax, ymax = convertBack(
            float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            #float(x), float(y), float(w), float(h))
        dets_tracker.append([xmin, ymin, xmax, ymax, detection[1]])
        dets_nms.append([x, y, w, h])
        dets_confidence.append(detection[1])
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets_tracker = np.asarray(dets_tracker)
    dets_nms = np.asarray(dets_nms)
    dets_confidence = np.asarray(dets_confidence)
    return dets_tracker, dets_nms, dets_confidence


netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3.cfg"
    weightPath = "./yolov3.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("test.mp4")
    W = int(cap.get(3))
    H = int(cap.get(4))
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'FLV1'),
            cap.get(cv2.CAP_PROP_FPS), (W, H))
    memory = {}

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                       darknet.network_height(netMain), 3)
    strt_time = time.time()
    print('Start Time', strt_time)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(
            netMain, metaMain, darknet_image, thresh=0.5)

        tracker_dets, nms_dets, dets_confidence = convertToDetectionFormat(detections, (W, H))

        indices = non_max_suppression(nms_dets, 0.75, dets_confidence)
        detections = [detections[i] for i in indices]

        tracker_dets = convertToDetectionFormat(detections, (W, H))
        tracks = tracker.update(tracker_dets)

        boxes = []
        indexIDs = []
        previous = memory.copy()
        memory = {}

        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]

        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                # draw a bounding box rectangle and label on the image
                # color = [int(c) for c in COLORS[classIDs[i]]]
                # cv2.rectangle(frame_read, (x, y), (x + w, y + h), color, 2)

                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                cv2.rectangle(frame_read, (x, y), (w, h), color, 2)

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                    p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                    cv2.line(frame_read, p0, p1, color, 3)

                # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                text = "{}".format(indexIDs[i])
                cv2.putText(frame_read, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1

        # image = cvDrawBoxes(detections, frame_read)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print('Execution Time Per Frame', time.time()-prev_time)
        print(1/(time.time()-prev_time))
        #cv2.imshow('Demo', image)
        out.write(frame_read)
        cv2.waitKey(3)
    print('End Time', time.time(), 'Elapsed Time', time.time() - strt_time)
    cap.release()
    out.release()


if __name__ == "__main__":
    YOLO()
