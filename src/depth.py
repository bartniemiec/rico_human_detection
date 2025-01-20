#!/usr/bin/env python
import rospy
import sys
from receive_image.msg import coordinates, results
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2


class DepthReader:
    def __init__(self):
        self.bridge = CvBridge()
        self.coordinates_sub = rospy.Subscriber("/coordinates", coordinates, self.coordinates_callback, queue_size=1)
        self.results_pub = rospy.Publisher("/results", results, queue_size=1)
        self.path = '/home/rico/Desktop/BartoszNiemiecHumanDetection/src/receive_image/include/receive_image/written_depth.jpg'

    def coordinates_callback(self, data):
        image = data.depth_image
        try:
            depth_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
            depth_image = cv2.resize(depth_image, (1080, 720))
            depth_array = np.array(depth_image, dtype=np.float32)
            cv2.circle(depth_image, (data.x, data.y), 10, (0, 255, 0), 5)
            cv2.imwrite(self.path, depth_image)
            if data.flag:
                msg = results()
                msg.is_human_detected = "YES"
                msg.distance = str(depth_array[data.y, data.x]/1000)
                self.results_pub.publish(msg)
            else:
                msg = results()
                msg.is_human_detected = "NO"
                msg.distance = "NONE"
                self.results_pub.publish(msg)
        except CvBridgeError as e:
 	        rospy.loginfo("There was an error converting ros message to image!")
            


def main(args):
    rospy.init_node('depth', anonymous=True)
    rospy.loginfo("Depth node created")
    dr = DepthReader()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down!")

if __name__ == '__main__':
    main(sys.argv)