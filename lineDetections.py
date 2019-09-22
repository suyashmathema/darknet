from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import csv

from utilities import *
from tracking import *
from lineSpeedEstimate import *

import argparse

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

netMain = None
metaMain = None
altNames = None


def YOLO(video='input.mp4', inputName="input", start=0, end=-1):
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

    cap = cv2.VideoCapture(video)
    width = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter(inputName+"-"+str(int(cap.get(cv2.CAP_PROP_FPS)))+"fps-line-detection.mp4", cv2.VideoWriter_fourcc(*'FLV1'),
                          cap.get(cv2.CAP_PROP_FPS), (width, height))

    outTracker = cv2.VideoWriter(inputName+"-"+str(int(cap.get(cv2.CAP_PROP_FPS)))+"fps-line-tracking.mp4", cv2.VideoWriter_fourcc(*'FLV1'),
                                 cap.get(cv2.CAP_PROP_FPS), (width, height))
    outSpeed = cv2.VideoWriter(inputName+"-"+str(int(cap.get(cv2.CAP_PROP_FPS)))+"fps-line-speed.mp4", cv2.VideoWriter_fourcc(*'FLV1'),
                               cap.get(cv2.CAP_PROP_FPS), (width, height))
    frame_count = 0
    sum_fps = 0

    init_speed_param(inputName)

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                       darknet.network_height(netMain), 3)
    strt_time = time.time()
    print('Start Time', strt_time)
    initialize_tracker()
    while True:
        if end > 0 and frame_count > end * cap.get(cv2.CAP_PROP_FPS):
            break
        prev_time = time.time()
        ret, frame_read = cap.read()
        if frame_count < start * cap.get(cv2.CAP_PROP_FPS):
            frame_count += 1
            continue
        if not ret:
            print('End of video, Exiting')
            break
        # Match color channel sequence, openCV use BGR and YOLO uses RGB
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        # Resizing image to match yolo input size
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        # yolo inferencing video frame
        detections = darknet.detect_image(
            netMain, metaMain, darknet_image, thresh=0.5)

        detections = convert_to_tracking_format(detections, (width, height))
        clone_frame = frame_read.copy()
        clone_frame_speed = frame_read.copy()

        detections, labels = filter_detections(detections)
        image = cvDrawBoxesLabel(detections, labels, frame_read)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        track_ids = tracking(detections, frame_count)
        imageTracked = cvDrawBoxesTracked(track_ids[:, :6], clone_frame)
        imageTracked = cv2.cvtColor(imageTracked, cv2.COLOR_BGR2RGB)

        if len(track_ids) > 0:
            vels = estimateSpeed(
                track_ids, frame_count, cap.get(cv2.CAP_PROP_FPS))
            imageSpeed = cvDrawBoxesSpeedLines(
                vels, track_ids[:, :6], getSpeedLines(), clone_frame_speed)
            imageSpeed = cv2.cvtColor(imageSpeed, cv2.COLOR_BGR2RGB)

        # print('Execution Time Per Frame', time.time()-prev_time)
        # print('fps', 1/(time.time()-prev_time), 'frame', frame_count)
        frame_count += 1
        sum_fps += (1/(time.time()-prev_time))
        if frame_count % 25 == 0:
            print(inputName+"-"+str(int(cap.get(cv2.CAP_PROP_FPS))),
                  "FPS:", sum_fps/25)
            sum_fps = 0

        out.write(frame_read)
        outTracker.write(clone_frame)
        outSpeed.write(clone_frame_speed)
        cv2.waitKey(0)
    print('End Time', time.time(), 'Elapsed Time', time.time() - strt_time)
    with open(inputName+"-"+str(int(cap.get(cv2.CAP_PROP_FPS)))+"fps-line-data.csv", mode='w') as csv_file:
        fieldnames = ["frame", "tid", "speed", "line1", "line2",
                      "xmin", "ymin", "xmax", "ymax", "score"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in getCsvData():
            writer.writerow(row)
    cap.release()
    out.release()
    outTracker.release()
    outSpeed.release()


if __name__ == "__main__":
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Path of the video')

    # Add the arguments
    my_parser.add_argument('-p',
                           metavar='path',
                           type=str,
                           help='the path to list')

    my_parser.add_argument('-s',
                           metavar='start',
                           type=int)

    my_parser.add_argument('-e',
                           metavar='end',
                           type=int)

    # Execute the parse_args() method
    args = my_parser.parse_args()

    input_path = args.p
    strt = args.s
    end = args.e
    name = input_path[0:input_path.rfind('-')]
    YOLO(input_path, name, strt, end)
