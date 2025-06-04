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
import threading
import Queue

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
        rospy.wait_for_service("detect")

        rgb_sub = Subscriber("/camera/color/image_raw", Image)
        depth_sub = Subscriber("/camera/depth/image_rect_raw", Image)
        ts = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=1, slop=1)
        ts.registerCallback(self.enqueue_latest_pair)

        self.latest_pair = Queue.Queue(maxsize=1)

        t = threading.Thread(target=self.processing_loop)
        t.daemon = True
        t.start()

    def enqueue_latest_pair(self, rgb, depth):
        # Replace previous pair with latest
        if self.latest_pair.full():
            try:
                self.latest_pair.get_nowait()
            except Queue.Empty:
                pass
        self.latest_pair.put_nowait((rgb, depth))

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

    def calc_params(self, rgb_image, depth_image):
        now = rospy.Time.now().to_sec()
        rgb_stamp = rgb_image.header.stamp.to_sec()
        depth_stamp = depth_image.header.stamp.to_sec()

        #LATENCY
        rospy.loginfo("RGB LATENCY: %s" % abs(rgb_stamp - now))
        rospy.loginfo("DEPTH LATENCY: %s" % abs(depth_stamp - now))

        #SYNC
        # if abs(rgb_stamp - depth_stamp) > 0.1:
        #     rospy.logwarn("SYNCHRONIZATION: %s" % abs(rgb_stamp - depth_stamp))
        # else:
        #     rospy.loginfo("SYNCHRONIZATION: %s" % abs(rgb_stamp - depth_stamp))

    def processing_loop(self):
        while not rospy.is_shutdown():
            try:
                rgb_msg, depth_msg = self.latest_pair.get(timeout=1)
                self.process_frame(rgb_msg, depth_msg)
            except Queue.Empty:
                continue

    def process_frame(self, rgb_image, depth_image):

        self.calc_params(rgb_image, depth_image)

        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_image, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logwarn("CV bridge conversion failed: %s", e)
            return

        t0 = time.time()
        # cv2.imwrite(self.path, cv_image)
        rospy.loginfo("Image saved in %.2f seconds", time.time() - t0)


        #call detection service and get response
        # try:
        #     response = self.detect_human()
        #     self.create_response(response)
        #     if response.flag:
        #         msg = self.create_message(depth_image, True)
        #         #send coordinates to depth node so to read the distance
        #         self.coordinates_pub.publish(msg)
        #         # rospy.loginfo("Human detected")
        #     else:
        #         msg = self.create_message(depth_image, False)
        #         #send coordinates to depth node so to read the distance
        #         self.coordinates_pub.publish(msg)
        #         # rospy.loginfo("No detection")
        # except rospy.ServiceException as e:
        #     print("Service call failed")


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
