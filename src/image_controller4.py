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
        self.rgb_latencies = []
        self.depth_latencies = []
        self.sync_offsets = []

        self.pipeline = rs.pipeline()
        self.init_timestamp = None
        self.init_rgb = None
        self.init_depth = None

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

    def calc_params(self, rgb_image, depth_image):
        now = (time.time() - self.init_timestamp)*1000
        rgb_stamp = rgb_image.get_timestamp() - self.init_rgb
        depth_stamp = depth_image.get_timestamp() - self.init_depth

        rgb_latency = abs(rgb_stamp - now)
        depth_latency = abs(depth_stamp - now)
        sync_offset = abs(rgb_image.get_timestamp() - depth_image.get_timestamp())

        self.rgb_latencies.append(rgb_latency)
        self.depth_latencies.append(depth_latency)
        self.sync_offsets.append(sync_offset)

        rospy.loginfo("RGB LATENCY: %s" % rgb_latency)
        rospy.loginfo("DEPTH LATENCY: %s" % depth_latency)

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

    def save_latency_plot(self):
        if len(self.timestamps) < 2:
            rospy.logwarn("Not enough data to plot latency/sync.")
            return

        times = [t - self.timestamps[0] for t in self.timestamps]
        print(times)

        plt.figure(figsize=(12, 6))

        avg_lat_rgb = np.mean(self.rgb_latencies)
        avg_lat_depth = np.mean(self.depth_latencies)

        plt.plot(times, self.rgb_latencies, label='RGB Latency (ms)', color='green')
        plt.plot(times, self.depth_latencies, label='Depth Latency (ms)', color='orange')
        plt.axhline(avg_lat_rgb, color='red', linestyle='--', label='Average RGB Latency: %.2f ms' % avg_lat_rgb)
        plt.axhline(avg_lat_depth, color='red', linestyle='--', label='Average Depth Latency: %.2f ms' % avg_lat_depth)

        plt.xlabel("Time (s)")
        plt.ylabel("Miliseconds")
        plt.title("Latency Over Time")
        plt.grid(True)
        plt.legend()

        plot_path = os.path.join(
            rospkg.RosPack().get_path('rico_human_detection'),
            'include', 'rico_human_detection', 'latency_plot_cable.png'
        )
        plt.savefig(plot_path)

    def save_sync_plot(self):
        if len(self.timestamps) < 2:
            rospy.logwarn("Not enough data to plot latency/sync.")
            return

        times = [t - self.timestamps[0] for t in self.timestamps]

        plt.figure(figsize=(12, 6))

        avg_sync = np.mean(self.sync_offsets[3:])

        plt.plot(times[3:], self.sync_offsets[3:], label='Sync Offset (ms)', color='purple')
        plt.axhline(avg_sync, color='red', linestyle='--', label='Average Sync Offset: %.2f ms' % avg_sync)

        plt.xlabel("Time (s)")
        plt.ylabel("Miliseconds")
        plt.title("Synchronization Over Time")
        plt.grid(True)
        plt.legend()

        plot_path = os.path.join(
            rospkg.RosPack().get_path('rico_human_detection'),
            'include', 'rico_human_detection', 'sync_plot_cable.png'
        )
        plt.savefig(plot_path)

    def capture_frames(self):
        if self.first:
            self.first = False
            frames = self.pipeline.wait_for_frames()
            rgb = frames.get_color_frame()
            depth = frames.get_depth_frame()
            self.init_rgb = rgb.get_timestamp()
            self.init_depth = depth.get_timestamp()
            self.init_timestamp = time.time()
            return None, None
        try:
            frames = self.pipeline.wait_for_frames()
            rgb = frames.get_color_frame()
            depth = frames.get_depth_frame()

            rgb_array = np.asanyarray(rgb.get_data())
            depth_array = np.asanyarray(depth.get_data())

            cv2.imwrite(self.path, rgb_array)
            if not rgb or not depth:
                rospy.logwarn("Incomplete frames received.")
                return None, None, None, None
            return rgb, depth
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
        
        self.calc_params(rgb_image, depth_image)

        current_time = time.time()

        depth_image = np.asanyarray(depth_image.get_data())

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

    def shutdown_hook():
        rospy.loginfo("Shutting down, saving frequency plot...")
        if ic.running:
            rospy.loginfo("Stopping RealSense pipeline...")
            ic.pipeline.stop()
            ic.running = False
        ic.save_frequency_plot()
        ic.save_latency_plot()
        ic.save_sync_plot()

    rospy.on_shutdown(shutdown_hook)

    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        rgb_image, depth_image = ic.capture_frames()
        if rgb_image is not None and depth_image is not None:
            ic.process_frame(rgb_image, depth_image)
        rate.sleep()

if __name__ == '__main__':
    main(sys.argv)
