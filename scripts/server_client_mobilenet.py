#!/usr/bin/python3.8
import rospy
import cv2
from std_msgs.msg import String
from receive_image.srv import detect, detectResponse
import math
import time
import tensorflow

class Service:
    def __init__(self):
        self.service = rospy.Service("detect", detect, self.service_callback)
        self.pub = rospy.Publisher('chatter', String, queue_size=1)
        self.x = None
        self.y = None

    def service_callback(self, req):
        model = tensorflow.keras.models.load_model('/root/bartosz_ws_3/src/receive_image/include/receive_image/model.h5')
        img = cv2.imread("/root/bartosz_ws_3/src/receive_image/include/receive_image/camera.jpg", 1)
        img_resize = cv2.resize(img, (224, 224))
        img_reshape = img_resize.reshape((1, 224, 224, 3))
        results = model.predict(img_reshape, verbose=0)
        rospy.loginfo(str(results))
        return detectResponse(x=self.x, y=self.y, name="test", prob=0.0, flag=True)

    

def detect_server():
    rospy.init_node("server_client")
    rospy.loginfo("Server node created")

    service = Service()

    # pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.spin()

if __name__ == "__main__":
    detect_server()