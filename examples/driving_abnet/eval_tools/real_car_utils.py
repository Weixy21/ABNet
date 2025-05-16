import os
import numpy as np
import utm
from tqdm import tqdm
import rosbag
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from descartes import PolygonPatch
from shapely import geometry
import pickle

import analysis_utils


def read_rosbag(bag_path, return_topic_info=False):
    bag = rosbag.Bag(bag_path)
    topic_info = bag.get_type_and_topic_info()[1]
    data = {k: [] for k in topic_info.keys()}
    for topic, msg, t in tqdm(bag.read_messages()):
        data[topic].append([t.to_sec(), msg])
    bag.close()

    if return_topic_info:
        return data, topic_info
    else:
        return data


def fetch_obstacle_pose(data, topic='/vista/obstacle/pose'):
    pose = []
    for t, msg in data[topic]:
        pose.append(msg.data)
    pose = np.array(pose)
    return pose


def fetch_gps(data, topic='/lexus/oxts/gps/fix'):
    gps= []
    for t, msg in data[topic]:
        x, y, _, _ = utm.from_latlon(msg.latitude, msg.longitude)
        gps.append([t, x, y])
    gps = np.array(gps)
    origin = gps[0,1:]
    return gps


def fetch_yaw(data, topic='/lexus/oxts/imu/data'):
    yaws = []
    for t, msg in data[topic]:
        q = msg.orientation
        yaw = np.arctan2(2.*(q.x*q.y + q.z*q.w) , (q.w**2 - q.z**2 - q.y*2 + q.x**2))
        yaws.append([t, yaw])
    yaws = np.array(yaws)
    return yaws


def plot_poly(ax, xs, ys, phis, color=[1., 0., 0.], alpha=0.2, label='label', dist_interval=5.5):
    # sample gps points to get equal interval across polygons
    if dist_interval is not None:
        dist = np.linalg.norm([xs[1:] - xs[:-1], ys[1:] - ys[:-1]], axis=0)
        mask = [True]
        cum_d = 0.
        for d in dist:
            cum_d += d
            if cum_d >= dist_interval:
                mask.append(True)
                cum_d = 0.
            else:
                mask.append(False)
        mask = np.array(mask)
        xs, ys = xs[mask], ys[mask],
        if phis.shape[0] < mask.shape[0]:
            phis = phis[mask[:phis.shape[0]]]
        else:
            d_len = phis.shape[0] - mask.shape[0]
            mask = np.concatenate([mask, np.array([False]*d_len, dtype=bool)])
            phis = phis[mask]

    # plot poly
    color = list(color) + [alpha]
    zorder = 10
    for x, y, phi in zip(xs, ys, phis):
        poly = analysis_utils.get_poly(np.array([x, y, phi]), [5., 2.])
        patch = PolygonPatch(poly, ec=color, fc=color, zorder=zorder)
        ax.add_patch(patch)
        zorder += 1

    patch_for_legend = mpatches.Patch(color=color[:3]+[alpha*2], label=label)

    return patch_for_legend


def load_devens_road(path):
    with open(path, 'rb') as f:
        loop_paths = pickle.load(f, encoding='latin1')

    return loop_paths


def plot_devens_road(loop_paths, figax=None, linewidth=0.25, color='k',
                     zorder=1, buffer=0, lns=dict()):
    if figax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig, ax = figax

    if buffer != 0:
        new_loop_paths = copy.deepcopy(loop_paths)

        for name, path in loop_paths.items():
            sign = -1 if "inner" in name else +1
            poly = geometry.Polygon(path).buffer(sign * buffer)
            new_loop_paths[name] = np.array(poly.exterior.coords.xy).T

        for name, path in new_loop_paths.items():
            ln_name = f'line:{name}'
            if ln_name in lns.keys():
                lns[ln_name].set_data(path[:,0], path[:,1])
            else:
                lns[ln_name], = ax.plot(path[:,0], path[:,1], linewidth=linewidth, color=color, zorder=zorder)
    else:
        for (name, path) in loop_paths.items():
            ln_name = f'line:{name}'
            if name in lns.keys():
                lns[ln_name].set_data(path[:,0], path[:,1])
            else:
                lns[ln_name], = ax.plot(path[:,0], path[:,1], linewidth=linewidth, color=color, zorder=zorder)
                ax.axis('equal')

    return [fig, ax], lns


def validate_path(path):
    valid_path = ['/'] if path.startswith('/') else []
    for v in path.split('/'):
        if v.startswith('$'):
            v = v[1:]
            assert v in os.environ, f'Remember to set ${v}'
            v = os.environ[v]
        valid_path.append(v)
    valid_path = os.path.join(*valid_path)
    valid_path = os.path.abspath(os.path.expanduser(valid_path))
    return valid_path
