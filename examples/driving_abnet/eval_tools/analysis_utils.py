import os
import pickle
import json
import numpy as np
from shapely.geometry import box as Box
from shapely import affinity


def read_results(results_path):
    with open(results_path, 'rb') as f:
        data = pickle.load(f)

    config_path = os.path.join(os.path.dirname(results_path), 'eval_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    return data, config


def compute_crashes(data):
    all_crashed = []
    for ep_data in data:
        has_crashed = False
        for step_data in ep_data:
            ego_agent_id = step_data['ego_agent_id']
            step_crashed = step_data['infos'][ego_agent_id]['crashed'] or (step_data['logs'][ego_agent_id]['d'] >= 2.)
            has_crashed = has_crashed or step_crashed
        all_crashed.append(has_crashed)
    return all_crashed


def compute_crash_rate(data):
    all_crashed = compute_crashes(data)
    return np.mean(all_crashed)


def compute_passes(data):
    all_passed = []
    for ep_data in data:
        has_passed = False
        for step_data in ep_data:
            ego_agent_id = step_data['ego_agent_id']
            step_passed = step_data['infos'][ego_agent_id]['n_passed'] > 0
            has_passed = has_passed or step_passed
        all_passed.append(has_passed)
    return all_passed


def compute_clearance(data):
    car_dim = np.array([5., 2.])
    all_clearance = []
    for ep_data in data:
        ep_clearance = []
        for step_data in ep_data:
            ego_agent_id = step_data['ego_agent_id']
            other_agent_id = [v for v in step_data['logs'].keys() if v != ego_agent_id][0]
            ego_agent_data = step_data['logs'][ego_agent_id]
            other_agent_data = step_data['logs'][other_agent_id]
            ego_poly = get_poly(np.array([ego_agent_data['x'], ego_agent_data['y'], ego_agent_data['yaw']]), car_dim)
            other_poly = get_poly(np.array([other_agent_data['x'], other_agent_data['y'], other_agent_data['yaw']]), car_dim)
            clearance = ego_poly.distance(other_poly)
            ep_clearance.append(clearance)
        all_clearance.append(ep_clearance)
    return all_clearance


def compute_minimal_clearance(data):
    all_clearance = compute_clearance(data)
    return np.mean([np.min(v) for v in all_clearance])


def get_poly(pose, car_dim):
    x, y, theta = pose
    car_length, car_width = car_dim
    poly = Box(x - car_width / 2., y - car_length / 2., x + car_width / 2.,
               y + car_length / 2.)
    poly = affinity.rotate(poly, np.degrees(theta))
    return poly


def compute_state_prediction_error(data, state_names=['ds', 'd', 'obs_d', 'mu'], avg_ep=True):
    all_err = {k: [] for k in state_names}
    for ep_data in data:
        for name in state_names:
            all_err[name].append([])

        for step_data in ep_data:
            ego_agent_id = step_data['ego_agent_id']
            for name in state_names:
                if name == 'dd':
                    pred_dd = step_data['logs'][ego_agent_id][f'model/dd']
                    pred_dd -= np.sign(pred_dd)
                    gt_dd = step_data['logs'][ego_agent_id][f'model/gt_dd']
                    gt_dd -= np.sign(gt_dd)
                    err = np.abs(pred_dd - gt_dd)
                else:
                    if f'model/pred_{name}' in step_data['logs'][ego_agent_id].keys():
                        pred = step_data['logs'][ego_agent_id][f'model/pred_{name}']
                    else:
                        pred = step_data['logs'][ego_agent_id][f'model/{name}']
                    gt = step_data['logs'][ego_agent_id][f'model/gt_{name}']
                    err = np.abs(pred - gt)
                all_err[name][-1].append(err)

        if avg_ep:
            for name in state_names:
                all_err[name][-1] = np.mean(all_err[name][-1])

    return all_err
