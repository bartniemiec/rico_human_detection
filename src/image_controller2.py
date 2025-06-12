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
import time
import rospkg
import os

import pyrealsense2 as rs
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

        # Camera setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)

        try:
            self.pipeline.start(self.config)
            time.sleep(2)  # Let camera warm up
            self.running = True
            rospy.loginfo("RealSense pipeline started successfully.")
        except Exception as e:
            rospy.logerr("Failed to start RealSense pipeline: %s", e)
            raise

    def shutdown(self):
        if self.running:
            rospy.loginfo("Stopping RealSense pipeline...")
            self.pipeline.stop()
            self.running = False

    def capture_frames(self):
        try:
            frames = self.pipeline.wait_for_frames()
            rgb = frames.get_color_frame()
            depth = frames.get_depth_frame()
            rgb_timestamp = rgb.get_timestamp()
            depth_timestamp = depth.get_timestamp()

            rgb_array = np.asanyarray(rgb.get_data())
            depth_array = np.asanyarray(depth.get_data())

            cv2.imwrite(self.path, rgb_array)
            if not rgb or not depth:
                rospy.logwarn("Incomplete frames received.")
                return None, None
            return rgb_array, rgb_timestamp, depth_array, depth_timestamp
        except Exception as e:
            rospy.logerr("Frame capture failed: %s", e)
            return None, None

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
        rospy.wait_for_service("detect")
        try:
            detect_human = rospy.ServiceProxy("detect",detect)
            response = detect_human()
            self.create_response(response)
            if response.flag:
                msg = self.create_message(self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1"), True)
                #send coordinates to depth node so to read the distance
                self.coordinates_pub.publish(msg)
                # rospy.loginfo("Human detected")
            else:
                msg = self.create_message(self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1"), False)
                #send coordinates to depth node so to read the distance
                self.coordinates_pub.publish(msg)
                # rospy.loginfo("No detection")
        except rospy.ServiceException as e:
            print("Service call failed")

        
def main(args):
    rospy.init_node('image_converter', anonymous=True)
    rospy.loginfo("View image node created")
    ic = ImageConverter()
    rospy.on_shutdown(ic.shutdown)

    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        start = time.time()
        rgb_image, rgb_timestamp, depth_image, depth_timestamp = ic.capture_frames()
        stop = time.time()
        print("RGB LATENCY: ", abs(rgb_timestamp - start)*1000, "ms")
        print("DEPTH LATENCY: ", abs(depth_timestamp - start)*1000, "ms")
        sync_diff = abs(rgb_timestamp - depth_timestamp)*1000
        # print("SYNC DIFF: ", sync_diff, "ms")
        if rgb_image is not None and depth_image is not None:
            # rospy.loginfo("Captured frame!")
            ic.process_frame(rgb_image, depth_image)
        rate.sleep()
        # stop = time.time()
        # print("TIME OF ITERATION: ", stop - start, "s")

if __name__ == '__main__':
    main(sys.argv)
