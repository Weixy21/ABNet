#!/usr/bin/env python
import time
import math

import deepknight
import numpy as np
import ros_numpy
import rospy
import utm
from collections import deque
from cv_bridge import CvBridge
from ethernet_interface.msg import UdpInMsg
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Image, NavSatFix, PointCloud2, Imu
from rospy.msg import AnyMsg
from std_msgs.msg import Float64
from prophesee_event_msgs.msg import Event, EventArray

from util import *

# deepknight_node
# Author: Alexander Amini - amini@mit.edu
# Inputs: sensor_msgs/Image - An image from a front-facing camera
# Outputs: std_msgs/Float64 - The desired tire angle (radians)


CONTROL_SPEED = False

@ros_numpy.registry.converts_to_numpy(EventArray)
def convert(msg):
    out_pos = []
    out_neg = []
    for event in msg.events:
        if event.polarity:
            out_pos.append([event.y, event.x])
        else:
            out_neg.append([event.y, event.x])
    return np.array(out_pos), np.array(out_neg)


class DeepknightNode(object):
    def __init__(self):

        # Initialize the model controller
        self.set_default_vars()
        model_type = rospy.get_param('~type')
        model_path = rospy.get_param('~model')
        map_path = rospy.get_param('~map')

        DeepknightController = deepknight.spawn(model_type)
        self.controller = DeepknightController(model_path=model_path,
                                               navigator_path=map_path)
        self.node_name = 'deepknight_node'
        self.br = CvBridge()

        # Setup the subscriber and publisher
        self.loginfo([v for v in rospy.get_published_topics() if "image" in v[0]])
        self.sub_image = rospy.Subscriber("camera_array/camera_front/image_raw_ar",
                                          Image,
                                          self.image_callback,
                                          queue_size=1,
                                          buff_size=695820800)
        self.sub_event = rospy.Subscriber("/prophesee/camera/event_frame",
                                          Image,
                                          self.event_callback,
                                          queue_size=1,
                                          buff_size=695820800)
        self.sub_odom = rospy.Subscriber("odometry/filtered/odom",
                                         Odometry,
                                         self.odom_callback,
                                         queue_size=1)
        self.sub_lidar = rospy.Subscriber("velodyne_points",
                                          PointCloud2,
                                          self.lidar_callback,
                                          queue_size=1)
        self.sub_gps = rospy.Subscriber("oxts/gps/fix",
                                          NavSatFix,
                                          self.gps_callback,
                                          queue_size=1)
        self.sub_gps = rospy.Subscriber("oxts/imu/data",
                                          Imu,
                                          self.imu_callback,
                                          queue_size=1)
        self.sub_speed = rospy.Subscriber("pacmod/as_tx/vehicle_speed",
                                          Float64,
                                          self.speed_callback,
                                          queue_size=1)
        self.sub_ethernet = rospy.Subscriber(
            "ros_udp_interface_node/ethernet_in",
            UdpInMsg,
            self.eth_callback,
            queue_size=1)
        self.pub = rospy.Publisher("deepknight/goal_steer",
                                   Float64,
                                   queue_size=1)
        if CONTROL_SPEED:
            self.pub_speed = rospy.Publisher("deepknight/goal_speed",
                                             Float64,
                                             queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1), self.timer_callback)

        self.loginfo("Started!")


    def set_default_vars(self):
        # Some variables for tracking the state of the vehicle
        self.fcamera = np.zeros((250, 400, 3), dtype=np.uint8)  # current image
        self.lidar = None  # current point cloud data
        self.latlon = (0.0, 0.0)  # latitude and longitude
        self.yaw = 0.0  # yaw heading
        self.pos_last = None  # position for computing elapsed distance
        self.elapsed_dist = 0.0  # elapsed distance since starting the car
        self.measured_curvature = 0.0  # current curvature of the car
        self.desired_curvature = 0.0  # desired/predicted curvature
        self.alpha = 0.0  # smoothing factor (0.0 = no smoothing)
        self.auto_mode = False  # if the car is in autonomous mode or not
        self.last_tic = time.time() # start the clock
        self.yaw_rate = 0.0
        self.speed = 0.0
        self.event_cam_size = (480, 640)
        self.event_camera = np.zeros((self.event_cam_size[0], self.event_cam_size[1], 3),
                                     dtype=np.uint8)

    def run_controller(self):
        """ Run the specified deepknight controller. The most recently obtained
        sensor readings will be  directly forwarded to the controller and
        the resulting curvature is returned and published."""

        self.position = (self.latlon[0], self.latlon[1], self.yaw)

        inverse_r = self.controller.forward(
                                        self.measured_curvature,
                                        self.speed,
                                        fcamera=self.fcamera,
                                        event_camera=self.event_camera,
                                        lidar=self.lidar,
                                        gps=self.position,
                                        current_dist=self.elapsed_dist)
        # inverse_r += -0.01 #-0.012 # DEBUG Event
        # inverse_r += -0.009 #-0.012 # DEBUG Event gopro
        # inverse_r += -0.008 # DEBUG lidar
        # inverse_r += -0.01 # DEBUG lidar
        self.desired_curvature = self.alpha * self.desired_curvature + \
                (1 - self.alpha) * inverse_r

        curvature_msg = self.make_float_msg(self.desired_curvature)
        self.pub.publish(curvature_msg)

        if CONTROL_SPEED:
            self.pub_speed.publish(self.controller.speed_control)

        event_frame = self.event_camera if self.controller.master_callback == \
            deepknight.Callback.EVENT else None
        self.controller.draw_gui(self.fcamera, self.desired_curvature,
                                 self.measured_curvature, event_frame)
        elapsed = time.time() - self.last_tic
        self.loginfo("rate: %.1f \t inverse_r: %.3f \t measured_curvature: %.3f" % (1./elapsed, self.desired_curvature, self.measured_curvature))
        self.last_tic = time.time()

    # A dummy test callback that runs at a fixed rate
    def timer_callback(self, timer): 
        if self.controller.master_callback == deepknight.Callback.TIMER:
            self.fcamera = np.zeros((200, 320, 3))
            self.run_controller()


    def image_callback(self, image_msg):
        self.fcamera = self.br.imgmsg_to_cv2(image_msg, 'bgr8')
        # self.fcamera = 255 - self.fcamera
        if False: # NOTE: temp for policy trained with multi-agent vista
            h, w, d = self.fcamera.shape
            shift_h, shift_w = 10, -40 #-15, -18
            padded_image = np.zeros((h+abs(shift_h), w+abs(shift_w), d), dtype=self.fcamera.dtype)
            #padded_image[shift_h:h+shift_h, shift_w:w+shift_w, :] = self.fcamera # :-15, 18:
            padded_image[:h, :w] = self.fcamera
            padded_image = np.roll(padded_image, shift_w, axis=1)
            padded_image = np.roll(padded_image, shift_h, axis=0)
            self.fcamera = padded_image[:h, :w, :d]

        if self.controller.master_callback == deepknight.Callback.IMAGE:
            self.run_controller()

    def event_callback(self, msg):
        self.event_camera = events2frame(self.br.imgmsg_to_cv2(msg, 'mono8'),
                                         self.event_cam_size[0],
                                         self.event_cam_size[1],
                                         is_frame=True)

        if self.controller.master_callback == deepknight.Callback.EVENT:
            self.run_controller()

    def lidar_callback(self, msg):
        lidar_np = ros_numpy.numpify(msg)

        self.lidar = np.array([
            lidar_np['x'], lidar_np['y'], lidar_np['z'], lidar_np['intensity']
        ]).T

        if self.controller.master_callback == deepknight.Callback.LIDAR:
            self.run_controller()

    def gps_callback(self, msg):
        self.latlon = (msg.latitude, msg.longitude)

    def imu_callback(self, msg):
        # Convert quaternion to yaw heading angle:
        # https://stackoverflow.com/a/18115837
        q = msg.orientation
        yaw_y = 2. * (q.x * q.y + q.z * q.w)
        yaw_x = q.w ** 2 - q.z ** 2 - q.y * 2 + q.x ** 2
        self.yaw = np.arctan2(yaw_y, yaw_x)

        self.yaw_rate = msg.angular_velocity.z


    def odom_callback(self, msg):
        vx = msg.twist.twist.linear.x
        if vx > 1e-4:
            self.measured_curvature = msg.twist.twist.angular.z / vx

        pos_now = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.pos_last is None:
            self.pos_last = pos_now

        dist = math.sqrt((pos_now[0] - self.pos_last[0]) ** 2 +
                         (pos_now[1] - self.pos_last[1]) ** 2)
        self.elapsed_dist += dist
        self.pos_last = pos_now

    def speed_callback(self, msg):
        vx = msg.data
        if vx > 1e-4:
            self.measured_curvature = self.yaw_rate / vx
        self.speed = vx

    def eth_callback(self, msg):
        if self.auto_mode != msg.auto_mode:
            self.controller.enter_auto()
            self.auto_mode = msg.auto_mode

    def make_float_msg(self, num):
        cmd_msg = Float64()
        cmd_msg.data = num
        return cmd_msg

    def loginfo(self, msg):
        rospy.loginfo("[%s] %s", self.node_name, msg)


def events2frame(events, cam_h, cam_w, positive_color=[255, 255, 255],
                 negative_color=[212, 188, 114], mode=2, is_frame=False):
    if is_frame:
        frame = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        frame[events == 255] = positive_color
        frame[events == 0] = negative_color
    else:
        if mode == 0:
            frame = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            for color, p_events in zip([positive_color, negative_color], events):
                uv = np.concatenate(p_events)[:, :2]
                frame[uv[:, 0], uv[:, 1], :] = color
        elif mode == 1:
            frame_acc = np.zeros((cam_h, cam_w), dtype=np.int8)
            for polarity, p_events in zip([1, -1], events):
                for sub_p_events in p_events:
                    uv = sub_p_events[:, :2]
                    frame_acc[uv[:, 0], uv[:, 1]] += polarity

            frame = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            frame[frame_acc > 0, :] = positive_color
            frame[frame_acc < 0, :] = negative_color
        elif mode == 2:
            frame_abs_acc = np.zeros((cam_h, cam_w), dtype=np.int8)
            frame = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            for polarity, p_events in zip([1, -1], events):
                for sub_p_events in p_events:
                    if sub_p_events.shape[0] == 0:
                        continue
                    uv = sub_p_events[:, :2]
                    add_c = np.array(
                        positive_color if polarity > 0 else negative_color)[None,
                                                                            ...]
                    import time
                    tic = time.time()
                    cnt = frame_abs_acc[uv[:, 0], uv[:, 1]][:, None]
                    toc = time.time()
                    print('1', toc-tic)

                    tic = time.time()
                    frame[uv[:, 0], uv[:, 1]] = (frame[uv[:, 0], uv[:, 1]] * cnt +
                                                add_c) / (cnt + 1)
                    toc = time.time()
                    print('2', toc-tic)

                    tic = time.time()
                    frame_abs_acc[uv[:, 0], uv[:, 1]] = cnt[:, 0] + 1
                    toc = time.time()
                    print('3', toc-tic)
        else:
            raise NotImplementedError('Unknown mode {}'.format(mode))
    return frame


if __name__ == '__main__':
    rospy.init_node('deepknight_node', anonymous=False)
    deepknight_node = DeepknightNode()
    rospy.spin()
