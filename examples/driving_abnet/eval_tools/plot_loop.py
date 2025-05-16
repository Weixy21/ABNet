import os
import numpy as np
import argparse
import utm
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from scipy.interpolate import interp1d
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

import analysis_utils as utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-paths', type=str, nargs='+', required=True)
    parser.add_argument('--labels', type=str, nargs='+', default=None)
    parser.add_argument('--trace-path', default=None)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--rotate-deg', type=float, default=-10.)
    parser.add_argument('--cbar-tick-size', type=float, default=18.)
    parser.add_argument('--title-size', type=float, default=24.)
    parser.add_argument('--cbar-font-size', type=float, default=24.)
    parser.add_argument('--scalebar-font-size', type=float, default=18.)
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [f'data_{i}' for i in range(len(args.results_paths))]

    data = dict()
    config = dict()
    for label, results_path in zip(args.labels, args.results_paths):
        data[label], config[label] = utils.read_results(results_path)

        # print(np.mean([len(v) for v in data[label]]))
        # if True:
        #     data[label] = [v for v in data[label] if len(v) >= 45]

    # get loop data from trace
    if args.trace_path is None:
        args.trace_path = config[args.labels[0]]['trace_paths'][0]

    gps = read_csv_np(os.path.join(args.trace_path, 'gps.csv'))
    gps_xy = np.array([utm.from_latlon(v[0], v[1])[:2] for v in gps[:, 1:3]])
    gps_xy -= gps_xy[0]
    camera_timestamps = read_csv_np(os.path.join(args.trace_path, 'camera_front.csv'))
    gps_x_fn = interp1d(gps[:, 0], gps_xy[:,0],fill_value='extrapolate')
    gps_y_fn = interp1d(gps[:, 0], gps_xy[:,1],fill_value='extrapolate')
    aligned_gps_xy = np.stack([gps_x_fn(camera_timestamps[:,1]), gps_y_fn(camera_timestamps[:,1])], axis=1)
    if args.rotate_deg != 0.:
        aligned_gps_xy = rotate_xy(aligned_gps_xy, np.radians(args.rotate_deg))

    # compute tracking error
    track_errs = dict()
    for label in args.labels:
        track_errs[label] = [[] for _ in range(camera_timestamps.shape[0])]
        for ep_data in data[label]:
            for step_data in ep_data:
                ego_agent_id = step_data['ego_agent_id']
                frame_index = step_data['logs'][ego_agent_id]['frame_index']
                dev = np.abs(step_data['logs'][ego_agent_id]['model/gt_d'])
                dev = (dev > 1) # lateral shift from lane center larger than 1m
                track_errs[label][frame_index].append(dev)
                if dev > 3.: # stop if too off the lane
                    break
        track_errs[label] = np.array([np.mean(v) if len(v) > 0 else 0. for v in track_errs[label]])

    fig, axes = plt.subplots(1, len(args.labels), figsize=(6*len(args.labels), 5))
    cmap = cm.get_cmap('viridis')
    vmin = min([v.min() for v in track_errs.values()])
    vmax = min([v.max() for v in track_errs.values()])
    cnorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    for i, label in enumerate(args.labels):
        axes[i].scatter(aligned_gps_xy[:,0], aligned_gps_xy[:,1], c=track_errs[label], s=30, cmap=cmap, norm=cnorm)
        axes[i].set_aspect('equal')

        scalebar = ScaleBar(0.8, location='lower center', font_properties={'size': args.scalebar_font_size})
        axes[i].add_artist(scalebar)

        axes[i].set_title(label, fontsize=args.title_size)

    for i, ax in enumerate(axes):
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        if i != len(axes) - 1:
            cax.set_xticks([])
            cax.set_yticks([])
            cax.axis('off')
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap), cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=args.cbar_tick_size)
    cbar.ax.get_yaxis().labelpad = 30
    cbar.ax.set_ylabel('P (Deviation > 1m)', rotation=270, fontsize=args.cbar_font_size)

    fig.tight_layout()

    if args.save_path:
        fig.savefig(args.save_path)


def rotate_xy(xy, theta):
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    center = xy.mean(0)
    new_xy = np.matmul(xy - center, rot.T) + center
    return new_xy


def read_csv_np(path):
    data = np.genfromtxt(path, delimiter=',')
    if np.isnan(data[0,0]):
        data = data[1:].copy()
    return data


if __name__ == '__main__':
    main()
