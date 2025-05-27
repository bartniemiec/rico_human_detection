#!/usr/bin/python
import cv2
import sys
from PIL import Image as im 
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rico_human_detection.msg import Coordinates, Results
from cv_bridge import CvBridge, CvBridgeError
from rico_human_detection.srv import detect, detectResponse
from message_filters import Subscriber, ApproximateTimeSynchronizer
import time
import rospkg
import os

class ImageConverter:
    def __init__(self):
        self.coordinates_pub = rospy.Publisher("/coordinates", Coordinates, queue_size=1)
        self.x = None
        self.y = None
        self.name = None
        self.confidence = None
        self.flag = None
        self.bridge = CvBridge()

        package_path = rospkg.RosPack().get_path('rico_human_detection')
        self.path = os.path.join(package_path, 'include', 'rico_human_detection', 'camera.jpg')

        rgb_sub = Subscriber("/camera/color/image_raw", Image)
        depth_sub = Subscriber("/camera/depth/image_rect_raw", Image)
        ts = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=1, slop=0.05)
        ts.registerCallback(self.process_frame)

    def create_message(self, depth_image, flag):
        msg = Coordinates()
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

    def process_frame(self, rgb_image, depth_image):

        now = rospy.Time.now().to_sec()
        rgb_stamp = rgb_image.header.stamp.to_sec()
        depth_stamp = depth_image.header.stamp.to_sec()

        #obliczanie latencji
        rgb_latency = (now - rgb_stamp)
        depth_latency = (now - depth_stamp)
        print("RGB latency: ", rgb_latency*1000, "ms", "Depth latency: ",  depth_latency*1000, "ms")

        #obliczanie sync
        time_diff = abs(rgb_stamp - depth_stamp)*1000
        print("Diff between frames: ", time_diff, "ms")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_image, desired_encoding="passthrough")
            cv_image = cv2.resize(cv_image, (1080, 720))
        except CvBridgeError as e:
            rospy.loginfo("There was an error converting ros message to image!")
        cv2.imwrite(self.path, cv_image)

        #call detection service and get response
        # rospy.loginfo("Waiting for service!")
        rospy.wait_for_service("detect")
        # rospy.loginfo("Wait is finally over!")
        try:
            detect_human = rospy.ServiceProxy("detect",detect)
            response = detect_human()
            self.create_response(response)
            if response.flag:
                msg = self.create_message(depth_image, True)
                #send coordinates to depth node so to read the distance
                self.coordinates_pub.publish(msg)
                # rospy.loginfo("Human detected")
            else:
                msg = self.create_message(depth_image, False)
                #send coordinates to depth node so to read the distance
                self.coordinates_pub.publish(msg)
                # rospy.loginfo("No detection")
        except rospy.ServiceException as e:
            print("Service call failed")

        
def main(args):
    rospy.init_node('image_converter', anonymous=True)
    rospy.loginfo("View image node created")
    ic = ImageConverter()
    try:
        # rospy.spin()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()
    except KeyboardInterrupt:
        print("Shutting down!")

if __name__ == '__main__':
    main(sys.argv)
