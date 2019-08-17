from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

from utilities import *
from sort import *

tracker = Sort()

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")


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

    cap = cv2.VideoCapture("test.mp4")
    width = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'FLV1'),
                          cap.get(cv2.CAP_PROP_FPS), (width, height))

    frame_count = 0

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
            print('End of video, Exiting')
            break
        frame_count += 1
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

        detections = convert_to_coord(detections, (width, height))
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # print('Execution Time Per Frame', time.time()-prev_time)
        print(1/(time.time()-prev_time))
        out.write(frame_read)
        cv2.waitKey(3)
    print('End Time', time.time(), 'Elapsed Time', time.time() - strt_time)
    cap.release()
    out.release()
