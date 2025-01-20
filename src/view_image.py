#!/usr/bin/python
import cv2
import sys
from PIL import Image as im 
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rico_human_detection.msg import coordinates, results
from cv_bridge import CvBridge, CvBridgeError
from rico_human_detection.srv import detect, detectResponse
import time

class ImageConverter:
    def __init__(self):
        self.init_pub = rospy.Publisher("/init_pub", String, queue_size=1)
        self.init_sub = rospy.Subscriber("/init_pub", String, self.image_callback, queue_size=1)
        self.coordinates_pub = rospy.Publisher("/coordinates", coordinates, queue_size=1)
        self.x = None
        self.y = None
        self.name = None
        self.confidence = None
        self.flag = None
        self.bridge = CvBridge()
        self.path = '/home/rico/Desktop/bartoszniemiec_ws/src/rico_human_detection/include/rico_human_detection/camera.jpg'

    def create_message(self, depth_image, flag):
        msg = coordinates()
        msg.x = self.x
        msg.y = self.y
        msg.depth_image = depth_image
        msg.flag = flag
        return msg
    
    def create_response(self, response):
        self.x = response.x
        self.y = response.y
        self.name = response.name
        self.confidence = response.prob
        self.flag = response.flag

    def image_callback(self, data):
        #collect frames and save it
        depth_image = rospy.wait_for_message("/camera/depth/image_rect_raw", Image)
        rgb_image = rospy.wait_for_message("/camera/color/image_raw", Image)

        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_image, desired_encoding="passthrough")
            cv_image = cv2.resize(cv_image, (1080, 720))
        except CvBridgeError as e:
            rospy.loginfo("There was an error converting ros message to image!")
        cv2.imwrite(self.path, cv_image)

        #call detection service and get response
        rospy.wait_for_service("detect")
        try:
            detect_human = rospy.ServiceProxy("detect",detect)
            response = detect_human()
            self.create_response(response)
            if response.flag:
                msg = self.create_message(depth_image, True)
                #send coordinates to depth node so to read the distance
                self.coordinates_pub.publish(msg)
            else:
                msg = self.create_message(depth_image, False)
                #send coordinates to depth node so to read the distance
                self.coordinates_pub.publish(msg)
                rospy.loginfo("No detection")
        except rospy.ServiceException as e:
            print("Service call failed")

        
def main(args):
    rospy.init_node('image_converter', anonymous=True)
    rospy.loginfo("View image node created")
    ic = ImageConverter()
    try:
        # rospy.spin()
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            ic.init_pub.publish("hi")
            rate.sleep()


    except KeyboardInterrupt:
        print("Shutting down!")

if __name__ == '__main__':
    main(sys.argv)
