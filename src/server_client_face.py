#!/usr/bin/python3.8
import rospy
import cv2
from std_msgs.msg import String
from ultralytics import YOLO
from receive_image.srv import detect, detectResponse
import math
import time

class Service:
    def __init__(self):
        self.model = YOLO("/root/bartosz_ws_3/src/receive_image/include/receive_image/yolov8n-face.pt")
        self.service = rospy.Service("detect", detect, self.service_callback)
        self.pub = rospy.Publisher('chatter', String, queue_size=1)
        self.x = None
        self.y = None

    def draw_binding_box(self, r, img):
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

        # put box in cam
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        self.x = int(round((x1+x2)/2))
        self.y = int(round(y1+y2)/2)
        cv2.circle(img, (self.x, self.y), 10, (255, 0, 0), 5)
        cv2.imwrite('written.jpg', img)
        

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

        cv2.putText(img, "face", org, font, fontScale, color, thickness)

    def service_callback(self, req):
        img = cv2.imread("/root/bartosz_ws_3/src/receive_image/include/receive_image/camera.jpg")
        results = self.model.predict(img,conf=0.5, verbose=True)
        try:
            for r in results:
                # self.draw_binding_box(r, img)
                detection_count = r.boxes
                for box in detection_count:
                    x1,y1,x2,y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    self.x = abs((x1-x2) / 2)
                    self.y = abs((y1-y2) / 2)
                    cv2.imwrite("written.jpg", img)
            self.pub.publish("Human detected")
            return detectResponse(x=self.x, y=self.y, name="face", prob=1, flag=True)
        except UnboundLocalError as e:
            self.pub.publish("No detection")
            rospy.loginfo("EXCEPTION")
            return detectResponse(x=None, y=None, name=None, prob=None, flag=False)

    

def detect_server():
    rospy.init_node("server_client")
    rospy.loginfo("Server node created")

    service = Service()

    # service = rospy.Service("detect", detect, service_callback)
    # pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.spin()

if __name__ == "__main__":
    detect_server()