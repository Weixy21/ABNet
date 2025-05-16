import argparse
from unittest.mock import patch
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib import cm
# from matplotlib_scalebar.scalebar import ScaleBar

import real_car_utils as utils


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag-paths', type=str, nargs='+', required=True)
    parser.add_argument('--labels', type=str, nargs='+', default=None)
    parser.add_argument('--align-obstacle', action='store_true')
    parser.add_argument('--devens-road', type=str, default=None)
    parser.add_argument('--devens-buffer', type=float, default=0.1)
    parser.add_argument('--devens-linewidth', type=float, default=1)
    parser.add_argument('--print-title', action='store_true')
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--tick-size', type=float, default=16)
    parser.add_argument('--unit-size', type=float, default=14)
    parser.add_argument('--legend-size', type=float, default=18)
    parser.add_argument('--xlim-margin', type=float, default=None)
    parser.add_argument('--ylim-margin', type=float, default=None)
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [f'data_{i}' for i in range(len(args.bag_paths))]

    # Extract data from bag
    data, gps, yaw, obstacle = dict(), dict(), dict(), dict()
    for label, bag_path in zip(args.labels, args.bag_paths):
        data[label] = utils.read_rosbag(bag_path)
        print('Topics')
        pprint(list(data[label].keys()))
        gps[label] = utils.fetch_gps(data[label])
        yaw[label] = utils.fetch_yaw(data[label])
        obstacle[label] = utils.fetch_obstacle_pose(data[label])

    # subtract by an offset
    xy_offset = gps[args.labels[0]][0, 1:].copy()
    for label in args.labels:
        gps[label][:, 1:] -= xy_offset
        obstacle[label][:, :2] -= xy_offset

    # align NOTE: no align yaw
    if args.align_obstacle:
        label_ref = args.labels[0]
        obs_ref = obstacle[label_ref][0:1]
        for label in args.labels:
            if label == label_ref:
                continue
            obs_diff = (obstacle[label][0] - obs_ref)[:,:2]
            obstacle[label][:,:2] -= obs_diff
            gps[label][:,1:] -= obs_diff

    for label in args.labels:
        obstacle[label][:, :2] -= 0.5

    # plot
    colors = list(cm.get_cmap('Set1').colors)[1:] + list(cm.get_cmap('Set2').colors)
    fig, ax = plt.subplots(1, 1)
    if args.print_title:
        title = []
        for bag_path in args.bag_paths:
            title.append(bag_path.split('/')[-2])
        title = '\n'.join(title)
        ax.set_title(title)
    legend_handles = []
    for i, label in enumerate(args.labels):
        ax.plot(gps[label][:,1], gps[label][:,2], label=label, c=colors[i])
        theta = -yaw[label][-1,1]
        dxy = np.array([np.sin(theta), np.cos(theta)]) * 0.5
        ax.arrow(gps[label][-1,1], gps[label][-1,2], dxy[0], dxy[1], color=colors[i],
                 shape='full', head_width=1, head_length=2)
        if not args.align_obstacle:
            ax.scatter(obstacle[label][0,0], obstacle[label][0,1], c='r', label=f'obstacle_{label}')

        patch_for_legend = utils.plot_poly(ax, gps[label][:, 1], gps[label][:, 2],
                                           yaw[label][:, 1], color=colors[i], label=label)
        legend_handles.append(patch_for_legend)

    if args.align_obstacle:
        i += 1
        patch_for_legend = utils.plot_poly(ax,
                                           obstacle[label_ref][0, 0:1],
                                           obstacle[label_ref][0, 1:2],
                                           obstacle[label_ref][0, 2:3],
                                           color=colors[i],
                                           label='Obstacle',
                                           dist_interval=None,
                                           alpha=0.3)
        legend_handles.append(patch_for_legend)

    ax.legend(handles=legend_handles, fontsize=args.legend_size)
    ax.tick_params(axis='x', labelsize=args.tick_size)
    ax.tick_params(axis='y', labelsize=args.tick_size)
    if args.xlim_margin is not None:
        args.xlim_margin = (ax.get_xlim()[1] - ax.get_xlim()[0]) * args.xlim_margin
        ax.set_xlim(ax.get_xlim()[0] - args.xlim_margin,
                    ax.get_xlim()[1] + args.xlim_margin)
    if args.ylim_margin is not None:
        args.ylim_margin = (ax.get_ylim()[1] - ax.get_ylim()[0]) * args.ylim_margin
        ax.set_ylim(ax.get_ylim()[0] - args.ylim_margin,
                    ax.get_ylim()[1] + args.ylim_margin)
    # ax.set_xlabel('(m)', loc='right', fontsize=args.unit_size)
    # scalebar = ScaleBar(1, location='lower right')  # 1 pixel = 1 meter
    # plt.gca().add_artist(scalebar)

    if args.devens_road:
        ori_xlim = ax.get_xlim()
        ori_ylim = ax.get_ylim()
        loop_paths = utils.load_devens_road(args.devens_road)
        loop_paths = {k: v - xy_offset for k, v in loop_paths.items()}
        utils.plot_devens_road(loop_paths, [fig, ax], linewidth=args.devens_linewidth, buffer=args.devens_buffer)
        ax.set_xlim(*ori_xlim)
        ax.set_ylim(*ori_ylim)

    fig.tight_layout()

    if args.save_path:
        fig.savefig(args.save_path)


if __name__ == '__main__':
    main()
