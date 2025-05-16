import argparse
import os
import numpy as np
import glob
import cv2

import real_car_utils as utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-bag-paths', nargs='+', required=True)
    parser.add_argument('--topic-name', type=str,
                        default='/lexus/camera_array/camera_front/image_raw_ar')
    parser.add_argument('--video-dir', type=str, required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.video_dir):
        os.makedirs(args.video_dir)

    for camera_bag_path in args.camera_bag_paths:
        data = utils.read_rosbag(camera_bag_path)
        video_path = os.path.join(args.video_dir, camera_bag_path.split('/')[-2]+'.mp4')
        if not dump_video(data, video_path, args.topic_name):
            print(f'No {args.topic_name} in {camera_bag_path}')
        else:
            print(f'Save video to {video_path}')


def ros_image_to_np(data):
    return np.frombuffer(data.data,
                         dtype=np.uint8).reshape(data.height, data.width, -1)


def dump_video(data, video_path, topic_name):
    if topic_name in data.keys():
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30,
                              (960, 600))
        for t, v in data[topic_name]:
            img = ros_image_to_np(v)
            out.write(img)
        out.release()
        return True
    else:
        return False


if __name__ == '__main__':
    main()
