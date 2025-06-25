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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

        self.timestamps = []
        self.frequencies = []
        self.rgb_latencies = []
        self.depth_latencies = []
        self.sync_offsets = []

        package_path = rospkg.RosPack().get_path('rico_human_detection')
        self.path = os.path.join(package_path, 'include', 'rico_human_detection', 'camera.jpg')
        self.results_path = os.path.join(package_path, 'include', 'rico_human_detection', 'plot.jpg')

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
        now = rospy.Time.now().to_sec()*1000
        rgb_stamp = rgb_image.header.stamp.to_sec()*1000
        depth_stamp = depth_image.header.stamp.to_sec()*1000

        rgb_latency = abs(rgb_stamp - now)
        depth_latency = abs(depth_stamp - now)
        sync_offset = abs(rgb_stamp - depth_stamp)

        self.rgb_latencies.append(rgb_latency)
        self.depth_latencies.append(depth_latency)
        self.sync_offsets.append(sync_offset)

        rospy.loginfo("RGB LATENCY: %s" % rgb_latency)
        rospy.loginfo("DEPTH LATENCY: %s" % depth_latency)

    def processing_loop(self):
        while not rospy.is_shutdown():
            try:
                rgb_msg, depth_msg = self.latest_pair.get(timeout=1)
                self.process_frame(rgb_msg, depth_msg)
            except Queue.Empty:
                continue

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

        plt.figure(figsize=(12, 6))

        avg_lat_rgb = np.mean(self.rgb_latencies)
        avg_lat_depth = np.mean(self.depth_latencies)


        plt.plot(times[:-3], self.rgb_latencies[:-3], label='RGB Latency (ms)', color='green')
        plt.plot(times[:-3], self.depth_latencies[:-3], label='Depth Latency (ms)', color='orange')
        plt.axhline(avg_lat_rgb, color='red', linestyle='--', label='Average RGB Latency: %.2f ms' % avg_lat_rgb)
        plt.axhline(avg_lat_depth, color='red', linestyle='--', label='Average Depth Latency: %.2f ms' % avg_lat_depth)

        plt.xlabel("Time (s)")
        plt.ylabel("Miliseconds")
        plt.title("Latency Over Time")
        plt.grid(True)
        plt.legend()

        plot_path = os.path.join(
            rospkg.RosPack().get_path('rico_human_detection'),
            'include', 'rico_human_detection', 'latency_plot.png'
        )
        plt.savefig(plot_path)

    def save_sync_plot(self):
        if len(self.timestamps) < 2:
            rospy.logwarn("Not enough data to plot latency/sync.")
            return

        times = [t - self.timestamps[0] for t in self.timestamps]

        plt.figure(figsize=(12, 6))

        avg_sync = np.mean(self.sync_offsets[3:])

        plt.plot(times, self.sync_offsets[1:], label='Sync Offset (ms)', color='purple')
        plt.axhline(avg_sync, color='red', linestyle='--', label='Average Sync Offset: %.2f ms' % avg_sync)

        plt.xlabel("Time (s)")
        plt.ylabel("Miliseconds")
        plt.title("Synchronization Over Time")
        plt.grid(True)
        plt.legend()

        plot_path = os.path.join(
            rospkg.RosPack().get_path('rico_human_detection'),
            'include', 'rico_human_detection', 'sync_plot.png'
        )
        plt.savefig(plot_path)

    def process_frame(self, rgb_image, depth_image):

        self.calc_params(rgb_image, depth_image)

        current_time = time.time()

        if self.timestamps:
            delta = current_time - self.timestamps[-1]
            freq = 1.0 / delta if delta > 0 else 0
        else:
            freq = 0.0  # First frame

        self.timestamps.append(current_time)
        self.frequencies.append(freq)

        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_image, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logwarn("CV bridge conversion failed: %s", e)
            return

        t0 = time.time()
        cv2.imwrite(self.path, cv_image)
        rospy.loginfo("Image saved in %.2f seconds", time.time() - t0)


        #call detection service and get response
        try:
            response = self.detect_human()
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

    def shutdown_hook():
        rospy.loginfo("Shutting down, saving frequency plot...")
        ic.save_frequency_plot()
        ic.save_latency_plot()
        ic.save_sync_plot()

    rospy.on_shutdown(shutdown_hook)

    try:
        # rospy.spin()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()
    except KeyboardInterrupt:
        print("Shutting down!")

if __name__ == '__main__':
    main(sys.argv)
