from roboflow import Roboflow
from ultralytics import YOLO
import cv2
import time
from PIL import Image
import warnings
import pyttsx3

alphabet = ['-','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']  
def detections():
        sentence = []
        MODEL = YOLO("C:/Users/johns/Downloads/best (1).pt")
        camera = cv2.VideoCapture(0)
        while True:
            ret, frame = camera.read()
            time.sleep(2)
            if not ret:
                break
        
            
            cv2.imwrite('img.jpg', frame)
            
            results = MODEL.predict(source = 'img.jpg',conf=.40, stream=False, show=True)
            try:
                for result in results:
                    if (result.boxes.cls).nelement() != 0:
                        sentence.append(alphabet[int(result.boxes.cls)])
                    else:
                        print("none detected")
            except ValueError:
                print("Error")
            except KeyboardInterrupt:
                break
            
        return sentence
def speak():
    sentence = detections()
    engine = pyttsx3.init()
    engine.say(sentence)
    engine.runAndWait()

warnings.simplefilter("ignore")
