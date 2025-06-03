#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import sys
import rospy
from sensor_msgs.msg import Image
from rico_human_detection.msg import Coordinates
from cv_bridge import CvBridge, CvBridgeError
from rico_human_detection.srv import detect
from message_filters import Subscriber, ApproximateTimeSynchronizer
from message_filters import TimeSynchronizer
from collections import deque
import rospkg
import threading
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

        self.detect_human = rospy.ServiceProxy("detect", detect)
        rospy.wait_for_service("detect")  # tylko raz

        self.rgb_queue = deque()
        self.depth_queue = deque()
        self.max_age = 0.2  # sekundy - maksymalny wiek ramek
        self.time_tolerance = 0.05
        self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)


    def rgb_callback(self, msg):
        now = rospy.Time.now().to_sec()
        rgb_time = msg.header.stamp.to_sec()
        if abs(now - rgb_time) < 0.7:
          self.rgb_queue.append(msg)
          self.try_process()
          return
        rospy.logwarn(abs(now - rgb_time))


    def depth_callback(self, msg):
        now = rospy.Time.now().to_sec()
        depth_time = msg.header.stamp.to_sec()
        if abs(now - depth_time) < 1.0:
          self.depth_queue.append(msg)
          self.try_process()
          return
        rospy.logwarn(abs(now - depth_time))



    def try_process(self):
        now = rospy.Time.now().to_sec()

        while self.rgb_queue and self.depth_queue:
            rgb_msg = self.rgb_queue[0]
            depth_msg = self.depth_queue[0]

            rgb_time = rgb_msg.header.stamp.to_sec()
            depth_time = depth_msg.header.stamp.to_sec()

            # Spróbuj zsynchronizować
            if abs(rgb_time - depth_time) < self.time_tolerance:
                self.rgb_queue.popleft()
                self.depth_queue.popleft()
                self.process_frame(rgb_msg, depth_msg)
                break  # Tylko jedna para na raz
            elif rgb_time < depth_time:
                self.rgb_queue.popleft()
            else:
                self.depth_queue.popleft()


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
            # cv_image = cv2.resize(cv_image, (1080, 720))
        except CvBridgeError as e:
            rospy.loginfo("There was an error converting ros message to image!")
        # cv2.imwrite(self.path, cv_image)

        #call detection service and get response
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
