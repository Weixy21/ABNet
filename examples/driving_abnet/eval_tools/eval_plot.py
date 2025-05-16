import os
import argparse
import importlib
import json
import pickle
import numpy as np
from skvideo.io import FFmpegWriter
import torch

from eval_tools import utils
import vista


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run VISTA for evaluation')
    # model
    parser.add_argument('--model-module',
                        type=str,
                        required=True,
                        help='Model class; should be consistent with the checkpoint')
    parser.add_argument('--ckpt',
                        type=str,
                        default=None,
                        help='Path to checkpoint')
    parser.add_argument('--state-net-model-module',
                        type=str,
                        default=None,
                        help='Model class for state net; should be consistent with the checkpoint')
    parser.add_argument('--state-net-ckpt',
                        type=str,
                        default=None,
                        help='Path to checkpoint for state net model')
    parser.add_argument('--ego-init-v',
                        type=float,
                        default=6.,
                        help='Initial velocity used for acceleration control')
    parser.add_argument('--set-obs-d-lower-bound',
                        type=float,
                        default=None,
                        help='Lower bound for predicted obs_d')
    parser.add_argument('--use-reference-control',
                        action='store_true',
                        default=False,
                        help='Use reference control')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='Not use cuda')
    # general vista arguments
    parser.add_argument('--trace-paths',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Path to the traces to use for simulation')
    parser.add_argument('--mesh-dir',
                        type=str,
                        default=None,
                        help='Directory of meshes for virtual agents')
    parser.add_argument('--n-agents',
                        type=int,
                        default=2,
                        help='Number of agents')
    parser.add_argument('--use-display',
                        action='store_true',
                        default=False,
                        help='Use VISTA default display')
    parser.add_argument('--road-width',
                        type=float,
                        default=4.,
                        help='Road width in VISTA')
    parser.add_argument('--reset-mode',
                        type=str,
                        default='default',
                        choices=['default', 'segment_start', 'uniform'],
                        help='Trace reset mode in VISTA')
    parser.add_argument('--use-curvilinear-dynamics',
                        action='store_true',
                        default=False,
                        help='Use curvilinear dynamics for vehicle dynamics')
    parser.add_argument('--max-step',
                        type=int,
                        default=None,
                        help='Maximal step to be ran')
    parser.add_argument('--init-dist-range',
                        type=float,
                        nargs='+',
                        default=[15., 25.],
                        help='Initial distance range of obstacle')
    parser.add_argument('--init-lat-noise-range',
                        type=float,
                        nargs='+',
                        default=[1., 1.5],
                        help='Initial lateral displacement of obstacle')
    # evaluation and logging
    parser.add_argument('--n-episodes',
                        type=int,
                        default=1,
                        help='Number of episodes')
    parser.add_argument('--out-dir',
                        type=str,
                        default=None,
                        help='Directory to save output')
    parser.add_argument('--save-video',
                        action='store_true',
                        default=False,
                        help='Save video for every episodes')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    USE_OLD_MODEL = False

    if USE_OLD_MODEL: # DEBUG
        import copy
        import torchvision
        # from old_models import CNNLSTM_FC as BNET
        from old_models import CNN_LSTM_DERI_BN as BNET
        conv_param = [[3, 24, 5, 2, 2], [24, 36, 5, 2, 2], [36, 48, 3, 2, 1],
                      [48, 64, 3, 1, 1], [64, 64, 3, 1, 1]]
        device = 'cuda' if not args.no_cuda else 'cpu'
        model = BNET(conv_param,
                     lstm_size=64,
                     q_size=2,
                     p_size=2,
                     dropout=0.3,
                     dev=device,).to(device)
                    #  with_final_layer=args.with_final_layer,
                    #  use_integration=args.use_integration,
                    #  use_derivative=args.use_derivative,
                    #  with_vel_delta_in=args.with_vel_delta_in,
                    #  add_lf_cbf=args.add_lf_cbf).to(device)

        ptv_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.3409, 0.3083, 0.2384],
                                             std=[0.2436, 0.1835, 0.1901])
        ])

        if args.ckpt:
            ckpt = torch.load(args.ckpt)
            model.load_state_dict(ckpt)
        model.eval()
        if not args.no_cuda:
            model.cuda()

        model.hparams = copy.deepcopy(args)
        model.hparams.use_fixed_standardize = True
        model.hparams.use_color_jitter = False

        # args.control_mode = 'delta-v'
        args.control_mode = 'omega-a'
        # model.hparams.output_mode = ['v', 'delta']
        model.hparams.output_mode = ['v', 'delta', 'a', 'omega']
    else:
        LitModel = importlib.import_module(f'.{args.model_module}', 'models').LitModel
        model = LitModel.load_from_checkpoint(args.ckpt)
        if set(model.hparams.output_mode) == set(['delta', 'v']):
            args.control_mode = 'delta-v'
        elif set(model.hparams.output_mode) == set(['omega', 'a']):
            args.control_mode = 'omega-a'
        elif set(model.hparams.output_mode[:4]) == set(['delta', 'v', 'omega', 'a']):
            args.control_mode = 'omega-a'
        else:
            raise NotImplementedError(f'No corresponding control mode for {model.hparams.output_mode}')
        if args.model_module == 'barrier_net':
            model_kwargs = {'solver': 'cvxpy',
                            'store_intermediate_data': True}
            model.hparams.not_use_gt = True
            if args.use_reference_control:
                model.hparams.model_type = 'deri_ref'
        elif args.model_module == 'old_barrier_net':
            model_kwargs = {'solver': 'cvxpy',
                            'store_intermediate_data': True}
            model.hparams.not_use_gt = True
            model.hparams.use_color_jitter = False
            model.hparams.use_fixed_standardize = True
        else:
            model_kwargs = dict()
        if not args.no_cuda:
            model.cuda()
        model.eval()

        if args.state_net_model_module:
            LitModelStateNet = importlib.import_module(f'.{args.state_net_model_module}', 'models').LitModel
            model_state_net = LitModelStateNet.load_from_checkpoint(args.state_net_ckpt)
            if not args.no_cuda:
                model_state_net.cuda()
            model_state_net.eval()
            model.hparams.use_indep_state_net = True
            model.hparams.indep_state_net_output = model_state_net.hparams.output_mode

    env = utils.get_env(args)
    if args.use_display:
        display_config = dict(road_buffer_size=1000, )
        display = vista.Display(env.world, display_config=display_config)
        display.reset()  # reset should be called after env reset
    ego_agent = env.world.agents[0]
    ego_agent._ego_dynamics._v = args.ego_init_v
    dt = 1 / 10.

    if args.out_dir:
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)

        config_path = os.path.join(args.out_dir, 'eval_config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=4)

        all_results = []

    stop_eval = False
    for episode_i in range(args.n_episodes):
        print(f'Episode {episode_i:04d}')
        observations = env.reset()
        if args.use_display:
            display.reset()  # reset should be called after env reset
        ego_agent._ego_dynamics._v = args.ego_init_v

        if args.out_dir:
            episode_results = []

        if args.save_video:
            assert args.out_dir is not None
            video_dir = os.path.join(args.out_dir, 'video')
            if not os.path.isdir(video_dir):
                os.makedirs(video_dir)
            video_path = os.path.join(video_dir, f'episode_{episode_i:04d}_without_Bnet.mp4')
            rate = f'{(1. / dt)}'
            video_writer = FFmpegWriter(video_path,
                                        inputdict={'-r': rate},
                                        outputdict={'-vcodec': 'libx264',
                                                    '-pix_fmt': 'yuv420p',
                                                    '-r': rate,})

        done = False
        if hasattr(model, 'get_initial_state'):
            rnn_state = model.get_initial_state(batch_size=1)
            if not args.no_cuda:
                rnn_state = [_v.cuda() for _v in rnn_state]
        else:
            rnn_state = None
        if args.state_net_model_module:
            if hasattr(model_state_net, 'get_initial_state'):
                rnn_state_state_net = model_state_net.get_initial_state(batch_size=1)
                if not args.no_cuda:
                    if isinstance(rnn_state_state_net, list):
                        rnn_state_state_net = [_v.cuda() for _v in rnn_state_state_net]
                    elif isinstance(rnn_state_state_net, dict):
                        for k, v in rnn_state_state_net.items():
                            assert isinstance(v, list)
                            rnn_state_state_net[k] = [vv.cuda() for vv in v]
                    else:
                        raise NotImplementedError
            else:
                rnn_state_state_net = None
        step_i = 0

        ego_x_bn = []
        ego_y_bn = []
        obs_x = []
        obs_y = []
        p1,p2 = [],[]
        ref_a, acc = [], []
        ref_omega, omega = [], []
        while not done:
            try:
                if USE_OLD_MODEL: # DEBUG
                    img_tensor = ptv_transform(
                        observations[ego_agent.id]['camera_front']
                        [:, :, ::-1].copy())[  # NOTE: to rgb
                            None, None,
                            ...].to(device)
                    roi = ego_agent.sensors[0].camera_param.get_roi()
                    img_tensor = img_tensor[..., roi[0]:roi[2], roi[1]:roi[3]]
                    model_inputs = utils.preprocess_obs(env, ego_agent, model, observations)
                    if not args.no_cuda:
                        model_inputs = [_v.cuda() for _v in model_inputs]
                    with torch.no_grad():
                        pred, rnn_state = model(img_tensor, model_inputs[1], model_inputs[2], 0, state_lstm=rnn_state)
                    pred = pred.cpu().numpy()[0]
                else:
                    model_inputs = utils.preprocess_obs(env, ego_agent, model, observations)
                    if not args.no_cuda:
                        model_inputs = [_v.cuda() for _v in model_inputs]
                    with torch.no_grad():
                        if args.state_net_model_module:
                            pred_state_net, rnn_state_state_net = model_state_net(model_inputs, rnn_state_state_net, **model_kwargs)
                            if args.set_obs_d_lower_bound is not None:
                                obs_d_idx = model_state_net.hparams.output_mode.index('obs_d')
                                if torch.sign(pred_state_net[:, :, obs_d_idx]) > 0:
                                    pred_state_net[:, :, obs_d_idx] = torch.clamp(
                                        pred_state_net[:, :, obs_d_idx], min=args.set_obs_d_lower_bound)
                                else:
                                    pred_state_net[:, :, obs_d_idx] = torch.clamp(
                                        pred_state_net[:, :, obs_d_idx], max=-args.set_obs_d_lower_bound)
                            if model_state_net.hparams.drop_obs_d_offset:
                                if 'obs_d' in model_state_net.hparams.output_mode:
                                    obs_d_idx = model_state_net.hparams.output_mode.index('obs_d')
                                    pred_state_net[:, :, obs_d_idx] += \
                                        torch.sign(pred_state_net[:, :, obs_d_idx]) * 5
                                if 'dd' in model_state_net.hparams.output_mode:
                                    dd_idx = model_state_net.hparams.output_mode.index('dd')
                                    pred_state_net[:, :, dd_idx] += \
                                        torch.sign(pred_state_net[:, :, dd_idx]) * 5
                            model_inputs.append(pred_state_net)
                        pred, rnn_state = model(model_inputs, rnn_state, **model_kwargs)
                    pred = pred.cpu().numpy()[0, 0] # drop batch and time dimension
                actions = utils.construct_actions(env, ego_agent, model, pred)

                observations, rewards, dones, infos = env.step(actions, dt)
                done = dones[ego_agent.id]

                if args.out_dir:
                    logs = dict()
                    for _a in env.world.agents:
                        if _a.id == ego_agent.id:
                            logs[_a.id] = utils.extract_logs(env, _a, model, pred)
                        else:
                            logs[_a.id] = utils.extract_logs(env, _a)
                    step_results = {
                        'ego_agent_id': ego_agent.id,
                        'logs': logs,
                        'actions': actions,
                        'infos': infos,
                    }
                    episode_results.append(step_results)

                if args.use_display:
                    img = display.render()
                    # img = utils.add_descriptions(env, ego_agent, model, pred, img)

                    if args.save_video:
                        video_writer.writeFrame(img)

                    if False: # DEBUG
                        # import cv2
                        # img_debug = observations[ego_agent.id]['camera_front']
                        # print(ego_agent.relative_state.x, actions)
                        # cv2.imwrite('tmp/test.png', img[:,:,::-1]) #_debug)
                        # import pdb; pdb.set_trace()
                        if step_i >= 100:
                            break
                        done = False # DEBUG dones[ego_agent.id]

                step_i += 1

                if args.max_step:
                    done = False # don't use terminal condition defined in env
                    if step_i >= args.max_step:
                        has_crashed = np.any([v['infos'][ego_agent.id]['crashed'] for v in episode_results])
                        print('n_passed: {}'.format(step_results['infos'][ego_agent.id]['n_passed'])
                            + ' has_crashed: {}'.format(has_crashed))
                        break
            except KeyboardInterrupt:
                if args.save_video:
                    video_writer.close()
                stop_eval = True
                break
            # except:
            #     break

        #     p1.append(model.p1.item())
        #     p2.append(model.p2.item())
        #     ref_a.append(model.ref_a.item())
        #     ref_omega.append(model.ref_omega.item())
        #     acc.append(pred[2].item())
        #     omega.append(pred[3].item())
        
        # p1_increase = np.array(p1)    
        # p1_increase.tofile('./plot/p1_increase.dat')
        # p2_increase = np.array(p2)    
        # p2_increase.tofile('./plot/p2_increase.dat')
        # ref_a_increase = np.array(ref_a)    
        # ref_a_increase.tofile('./plot/ref_a_increase.dat')
        # ref_omega_incease = np.array(ref_omega)    
        # ref_omega_incease.tofile('./plot/ref_omega_incease.dat')
        # a_increase = np.array(acc)    
        # a_increase.tofile('./plot/a_increase.dat')
        # omega_incease = np.array(omega)    
        # omega_incease.tofile('./plot/omega_incease.dat')

        #     ego_x_bn.append(env.world.agents[0].ego_dynamics.x)
        #     ego_y_bn.append(env.world.agents[0].ego_dynamics.y)
        # obs_x.append(env.world.agents[1].ego_dynamics.x)
        # obs_y.append(env.world.agents[1].ego_dynamics.y)
        
        # ego_x_bn = np.array(ego_x_bn)    
        # ego_x_bn.tofile('./plot/ego_x_bn1.dat')
        # ego_y_bn = np.array(ego_y_bn)    
        # ego_y_bn.tofile('./plot/ego_y_bn1.dat')
        # obs_x = np.array(obs_x)    
        # obs_x.tofile('./plot/obs_x.dat')
        # obs_y = np.array(obs_y)    
        # obs_y.tofile('./plot/obs_y.dat')

        if args.out_dir:
            all_results.append(episode_results)

        if args.save_video:
            video_writer.close()
        if stop_eval:
            break

    if args.out_dir:
        results_path = os.path.join(args.out_dir, 'results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(all_results, f)


if __name__ == '__main__':
    main()
