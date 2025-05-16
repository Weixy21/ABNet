import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import analysis_utils as utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-paths',
                        nargs='+',
                        required=True,
                        help='Path to results pkl')
    parser.add_argument('--labels',
                        nargs='+',
                        default=None,
                        help='Labels to each results')
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--axis-size', type=float, default=18)
    parser.add_argument('--legend-size', type=float, default=18)
    parser.add_argument('--tick-size', type=float, default=16)
    parser.add_argument('--linewidth', type=float, default=3)
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [f'data_{i}' for i in range(len(args.results_paths))]

    data = dict()
    config = dict()
    for label, results_path in zip(args.labels, args.results_paths):
        data[label], config[label] = utils.read_results(results_path)
        data[label] = [v for v in data[label] if len(v) >= 95]

    clearance = dict()
    has_crashed = dict()
    has_passed = dict()
    recall_data = dict()
    for label in args.labels:
        clearance[label] = utils.compute_clearance(data[label])

        has_crashed[label] = utils.compute_crashes(data[label])
        has_passed[label] = utils.compute_passes(data[label])

        clearance[label] = [v for i, v in enumerate(clearance[label]) if not has_crashed[label][i] and has_passed[label][i]]
        # clearance[label] = [v for i, v in enumerate(clearance[label]) if has_passed[label][i]]
        # clearance[label] = [v for i, v in enumerate(clearance[label]) if not has_crashed[label][i]]
        # clearance[label] = [vv for v in clearance[label] for vv in v]
        clearance[label] = [np.min(v) for v in clearance[label]]

        cl_sorted = sorted(clearance[label])
        x_data = np.array(cl_sorted)
        y_data = np.arange(x_data.shape[0]) / x_data.shape[0]
        recall_data[label] = [x_data, y_data]

    fig, ax = plt.subplots(1, 1)
    for k, (x_data, y_data) in recall_data.items():
        non_zero = x_data != 0
        x_data = x_data[non_zero]
        y_data = y_data[non_zero]
        ax.plot(x_data, y_data, label=k, linewidth=args.linewidth)
    ax.set_ylim(0., 1.1)
    ax.legend(loc='lower right', fontsize=args.legend_size)
    ax.set_xlabel(r'Clearance Threshold ($\lambda$)', fontsize=args.axis_size)
    ax.set_ylabel(r'P (clearance $>$ $\lambda$)', fontsize=args.axis_size)
    ax.tick_params(axis='x', labelsize=args.tick_size)
    ax.tick_params(axis='y', labelsize=args.tick_size)
    fig.tight_layout()

    if args.save_path:
        fig.savefig(args.save_path)


if __name__ == '__main__':
    main()
