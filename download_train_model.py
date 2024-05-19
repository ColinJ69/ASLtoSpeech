from ultralytics import YOLO
from roboflow import Roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="xxxxxxxxxxxxxx")
project = rf.workspace("221565zcv").project("cv2-a4ryn")
version = project.version(9)
dataset = version.download("yolov8")

data = f'{dataset.location}/data.yaml'

!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=6 imgsz=640
#I ran this code in google colab but idk how to put jupyter notebook on github nor do I want to learn how to.
