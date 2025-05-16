import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
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
    parser.add_argument('--hist-bins', nargs='+', default=[20, 20, 12, 12])
    parser.add_argument('--state-names', nargs='+', default=['d', 'mu', 'ds', 'obs_d'])
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--title-size', type=float, default=14)
    parser.add_argument('--ylabel-size', type=float, default=14)
    parser.add_argument('--tick-size', type=float, default=12)
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [f'data_{i}' for i in range(len(args.results_paths))]

    # read data
    data = dict()
    config = dict()
    for label, results_path in zip(args.labels, args.results_paths):
        data[label], config[label] = utils.read_results(results_path)

    state_errs = dict()
    crashes = dict()
    for label in args.labels:
        state_errs[label] = utils.compute_state_prediction_error(data[label], state_names=args.state_names)
        crashes[label] = utils.compute_crashes(data[label])

    # combine data
    state_err_mixed = {k: [] for k in args.state_names}
    crashes_mixed = []
    for label in args.labels:
        for name in args.state_names:
            state_err_mixed[name].extend(state_errs[label][name])
        crashes_mixed.extend(crashes[label])

    # compute histogram
    hist_crashes = dict()
    hist_bins = dict()
    crashes_mixed = np.array(crashes_mixed, dtype=bool)
    for i, name in enumerate(state_err_mixed.keys()):
        state_err_mixed[name] = np.array(state_err_mixed[name])
        hist_idcs, bins = get_hist_idcs(state_err_mixed[name], args.hist_bins[i])
        for i, idcs in enumerate(hist_idcs):
            if name not in hist_crashes.keys():
                hist_crashes[name] = [[] for _ in range(len(hist_idcs))]
            hist_crashes[name][i].extend(crashes_mixed[idcs])
        hist_bins[name] = bins

    # for label in args.labels:
    #     state_err = state_errs[label]
    #     crash = np.array(crashes[label])

    #     for name in state_err.keys():
    #         state_err[name] = np.array(state_err[name])
    #         hist_idcs = get_hist_idcs(state_err[name], args.hist_bins)
    #         for i, idcs in enumerate(hist_idcs):
    #             if name not in hist_crash.keys():
    #                 hist_crash[name] = [[] for _ in range(len(hist_idcs))]
    #             hist_crash[name][i].extend(crash[idcs])

    # plot
    name_map = {
        'd': r'$d$ (m)',
        'ds': r'$\Delta s$ (m)',
        'mu': r'$\mu$ (rad)',
        'obs_d': r'$d_{obs}$ (m)'
    }
    fig, axes = plt.subplots(2, 2, figsize=(9,6))
    axes = axes.flatten()
    for ax, name in zip(axes, args.state_names):
        w = (hist_bins[name][1:] - hist_bins[name][:-1]).mean()
        xs = (hist_bins[name][1:] + hist_bins[name][:-1]) / 2.
        ys = [np.mean(v) for v in hist_crashes[name]]
        mask = np.array([len(v) > 40 for v in hist_crashes[name]], dtype=bool)
        xs = np.array(xs)[mask]
        ys = np.array(ys)[mask]
        ax.bar(hist_bins[name][:-1][mask], ys, w, align='edge')

        ax.set_title('Error ' + name_map[name], fontsize=args.title_size)

        ax.set_ylabel(f'Crash Rate', fontsize=args.ylabel_size)
        ax.tick_params(axis='x', labelsize=args.tick_size)
        ax.tick_params(axis='y', labelsize=args.tick_size)

    fig.tight_layout()

    if args.save_path:
        fig.savefig(args.save_path)


def get_hist_idcs(vals, nbins):
    xs = np.arange(vals.shape[0])
    bins = np.linspace(vals.min(), vals.max(), nbins + 1)
    ind = np.digitize(vals, bins, right=True)
    result = [xs[ind == j] for j in range(1, nbins + 1)]
    return result, bins


if __name__ == '__main__':
    main()
