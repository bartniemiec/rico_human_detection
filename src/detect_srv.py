#!/usr/bin/python3.8
import rospy
import cv2
from std_msgs.msg import String
from ultralytics import YOLO
from rico_human_detection.srv import detect, detectResponse
import rospkg
import math
import time
import os

class Service:
    def __init__(self):
        package_path = rospkg.RosPack().get_path('rico_human_detection')
        self.model = YOLO(os.path.join(package_path, 'include', 'rico_human_detection', 'yolov8n.onnx'))

        self.service = rospy.Service("detect", detect, self.service_callback)
        self.pub = rospy.Publisher('chatter', String, queue_size=1)
        self.x = None
        self.y = None
        self.path = os.path.join(package_path, 'include', 'rico_human_detection', 'camera.jpg')

    def draw_binding_box(self, r, classNames, img):
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

        # put box in cam
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        self.x = int(round((x1+x2)/2))
        self.y = int(round(y1+y2)/2)
        cv2.circle(img, (self.x, self.y), 10, (255, 0, 0), 5)
        # cv2.imwrite('/home/rico/Desktop/BartoszNiemiecHumanDetection/src/rico_human_detection/include/rico_human_detection/written.jpg', img)
        

        # confidence
        confidence = math.ceil((box.conf[0]*100))/100
        # print("Confidence --->",confidence)

        # class name
        cls = int(box.cls[0])
        # print("Class name -->", classNames[cls])

        # object details
        org = [x1, y1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    def service_callback(self, req):
        classNames = ["human"]
        img = cv2.imread(self.path)

        results = self.model.predict(img, stream=True, classes=[0], conf=0.5, verbose=False)
        try:
            for r in results:
                self.draw_binding_box(r, classNames, img)
                detection_count = r.boxes.shape[0]
                for i in range(detection_count):
                    cls = int(r.boxes.cls[i].item())
                    name = r.names[cls]
                    confidence = float(r.boxes.conf[i].item())
            return detectResponse(x=self.x, y=self.y, name=name, prob=confidence, flag=True)
        except UnboundLocalError as e:
            return detectResponse(x=None, y=None, name=None, prob=None, flag=False)

    

def detect_server():
    rospy.init_node("server_client")
    rospy.loginfo("Server node created")

    service = Service()

    rospy.spin()

if __name__ == "__main__":
    detect_server()