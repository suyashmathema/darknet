from detections import *

vid_names = [
    'GOPR1485-Bike-30-25fps.mp4',
    'GOPR1485-Bike-30-30fps.mp4',
    'GOPR1485-Bike-30-50fps.mp4',
    'GOPR1486-Bike-40-25fps.mp4',
    'GOPR1486-Bike-40-30fps.mp4',
    'GOPR1486-Bike-40-50fps.mp4',
    'GOPR1487-Bike-50-25fps.mp4',
    'GOPR1487-Bike-50-30fps.mp4',
    'GOPR1487-Bike-50-50fps.mp4',
    'GOPR1488-Bike-50-25fps.mp4',
    'GOPR1488-Bike-50-30fps.mp4',
    'GOPR1488-Bike-50-50fps.mp4',
    'GOPR1489-Bike-60-25fps.mp4',
    'GOPR1489-Bike-60-30fps.mp4',
    'GOPR1489-Bike-60-60fps.mp4',
    'GOPR1489-Bike-60-90fps.mp4',
    'GOPR1489-Bike-60-120fps.mp4',
    'GOPR1491-Car-30-25fps.mp4',
    'GOPR1491-Car-30-30fps.mp4',
    'GOPR1491-Car-30-50fps.mp4',
    'GOPR1492-Car-40-25fps.mp4',
    'GOPR1492-Car-40-30fps.mp4',
    'GOPR1492-Car-40-50fps.mp4',
    'GOPR1493-Car-50-25fps.mp4',
    'GOPR1493-Car-50-30fps.mp4',
    'GOPR1493-Car-50-50fps.mp4',
    'GOPR1494-Car-50-25fps.mp4',
    'GOPR1494-Car-50-30fps.mp4',
    'GOPR1494-Car-50-50fps.mp4',
    'GOPR1495-Car-50-25fps.mp4',
    'GOPR1495-Car-50-30fps.mp4',
    'GOPR1495-Car-50-60fps.mp4',
    'GOPR1495-Car-50-90fps.mp4',
    'GOPR1495-Car-50-120fps.mp4',
    'GOPR1496-Car-60-25fps.mp4',
    'GOPR1496-Car-60-30fps.mp4',
    'GOPR1496-Car-60-60fps.mp4',
    'GOPR1496-Car-60-90fps.mp4',
    'GOPR1496-Car-60-120fps.mp4'
]

for vid in vid_names:
    name = vid[0:vid.rfind('-')]
    YOLO(vid, name)
