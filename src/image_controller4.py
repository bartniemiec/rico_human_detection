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
import matplotlib.pyplot as plt
import time
import rospkg
import threading
import Queue
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

        self.results_path = os.path.join(package_path, 'include', 'rico_human_detection', 'plot_cable.jpg')

        self.timestamps = []
        self.frequencies = []

        self.pipeline = rs.pipeline()

        t = threading.Thread(target=self.stream_camera)
        t.daemon = True
        t.start()

        # t2 = threading.Thread(target=self.enqueue_latest_pair)
        # t2.daemon = True
        # t2.start()

        self.latest_pair = Queue.Queue(maxsize=1)

        self.first = True


    def stream_camera(self):

        config = rs.config()
        self.running = None
        config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)

        try:
            self.pipeline.start(config)
            self.running = True
            rospy.loginfo("RealSense pipeline started successfully.")
        except Exception as e:
            rospy.logerr("Failed to start RealSense pipeline: %s", e)
            raise

    def enqueue_latest_pair(self, rgb, depth):
        # Replace previous pair with latest
        if self.latest_pair.full():
            try:
                self.latest_pair.get_nowait()
            except Queue.Empty:
                pass
        self.latest_pair.put_nowait((rgb, depth))

    def shutdown(self):
        if self.running:
            rospy.loginfo("Stopping RealSense pipeline...")
            self.pipeline.stop()
            self.running = False
        # self.save_frequency_plot()

    def calc_params(self, rgb_timestamp, depth_timestamp):
        now = rospy.Time.now().to_sec()
        rgb_stamp = rgb_timestamp * 1000
        depth_stamp = depth_timestamp * 1000

        #LATENCY
        rospy.loginfo("RGB LATENCY: %s" % abs(rgb_stamp - now))
        rospy.loginfo("DEPTH LATENCY: %s" % abs(depth_stamp - now))

        #SYNC
        # if abs(rgb_stamp - depth_stamp) > 0.1:
        #     rospy.logwarn("SYNCHRONIZATION: %s" % abs(rgb_stamp - depth_stamp))
        # else:
        #     rospy.loginfo("SYNCHRONIZATION: %s" % abs(rgb_stamp - depth_stamp))

    def save_frequency_plot(self):
        if len(self.timestamps) < 2:
            rospy.logwarn("Not enough data to plot frequency.")
            return

        # Compute relative time points aligned with frequency samples
        times = [t - self.timestamps[0] for t in self.timestamps[1:]]
        freqs = self.frequencies[:len(times)]

        # Apply moving average smoothing
        def moving_average(data, window_size=5):
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

        smoothed_freqs = moving_average(freqs, window_size=5)
        smoothed_times = times[:len(smoothed_freqs)]

        avg_freq = np.mean(smoothed_freqs)

        # Plot
        plt.figure()
        plt.plot(smoothed_times, smoothed_freqs, label='Smoothed Frequency (Hz)', color='blue')
        plt.axhline(avg_freq, color='red', linestyle='--', label='Average Hz: %.2f' % avg_freq)
        plt.xlabel("Time (s)")
        plt.ylabel("Hz")
        plt.title("Smoothed Processing Frequency Over Time")
        plt.grid(True)
        plt.legend()

        # Save plot
        plot_path = os.path.join(
            rospkg.RosPack().get_path('rico_human_detection'),
            'include', 'rico_human_detection', 'processing_frequency.png'
        )
        plt.savefig(self.results_path)

    def capture_frames(self):
        if self.first:
            self.first = False
            time.sleep(2)
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
                return None, None, None, None
            return rgb_array, depth_array, rgb_timestamp, depth_timestamp
        except Exception as e:
            rospy.logerr("Frame capture failed: %s", e)
            return None, None, None, None

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

        current_time = time.time()

        if self.timestamps:
            delta = current_time - self.timestamps[-1]
            freq = 1.0 / delta if delta > 0 else 0
        else:
            freq = 0.0  # First frame

        self.timestamps.append(current_time)
        self.frequencies.append(freq)

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
        rgb_image, depth_image, rgb_timestamp, depth_timestamp = ic.capture_frames()
        stop = time.time()
        # print("RGB LATENCY: ", abs(stop - start)*1000, "ms")
        # print("DEPTH LATENCY: ", abs(stop - start)*1000, "ms")
        print("SYNCHRONIZATION: ", abs(rgb_timestamp - depth_timestamp)*1000, "ms")
        if rgb_image is not None and depth_image is not None:
            ic.process_frame(rgb_image, depth_image)
        rate.sleep()

if __name__ == '__main__':
    main(sys.argv)
