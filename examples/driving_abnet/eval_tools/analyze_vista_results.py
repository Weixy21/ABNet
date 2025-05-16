import argparse
import numpy as np
from pprint import pprint

import analysis_utils as utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-path', required=True)
    args = parser.parse_args()

    data, config = utils.read_results(args.results_path)

    # if True:
    #     data = [v for v in data if len(v) >= 95]
    n_episodes = len(data)
    avg_steps = np.mean([len(v) for v in data])
    crash_rate = utils.compute_crash_rate(data)
    # minimal_clearance = utils.compute_minimal_clearance(data)

    # print('Barrier net path: {}'.format(config['ckpt']))
    # print('State net path: {}'.format(config['state_net_ckpt']))
    # print(f'Number of episodes: {n_episodes}')
    # print(f'Average steps: {avg_steps}')
    # print(f'Crash rate: {crash_rate}')
    # print(f'Minimal clearance: {minimal_clearance}')

    if  True:#crash_rate <= 1.0:
        print(f'N episodes: {n_episodes} Avg steps: {avg_steps} Crash rate: {crash_rate:.2f} ', end='')
        # print(f'Minimal clearance: {minimal_clearance:.2f} ', end='')
        # print('Bnet path: {} '.format('/'.join(config['ckpt'].split('/')[-4:])), end='')
        # print('Snet path: {}'.format('/'.join(config['state_net_ckpt'].split('/')[-4:])), end='')
        print(f'{args.results_path}')
    # bash eval_tools/batch_analyze.sh ~/mnt/robosim/results/bnet/eval_search/


if __name__ == '__main__':
    main()
