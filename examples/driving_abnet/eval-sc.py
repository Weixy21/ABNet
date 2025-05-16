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
    parser.add_argument('--ckpt-sc',
                        type=str,
                        default=None,
                        help='Path to checkpoint')
    parser.add_argument('--ckpt-sc3',
                        type=str,
                        default=None,
                        help='Path to checkpoint')
    parser.add_argument('--ckpt-sc4',
                        type=str,
                        default=None,
                        help='Path to checkpoint')
    parser.add_argument('--ckpt-sc5',
                        type=str,
                        default=None,
                        help='Path to checkpoint')
    parser.add_argument('--ckpt-sc6',
                        type=str,
                        default=None,
                        help='Path to checkpoint')
    parser.add_argument('--ckpt-sc7',
                        type=str,
                        default=None,
                        help='Path to checkpoint')
    parser.add_argument('--ckpt-sc8',
                        type=str,
                        default=None,
                        help='Path to checkpoint')
    parser.add_argument('--ckpt-sc9',
                        type=str,
                        default=None,
                        help='Path to checkpoint')
    parser.add_argument('--ckpt-sc10',
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
    parser.add_argument('--lf-cbf-threshold',
                        type=float,
                        default=2.,
                        help='Threshold for lane following CBF')
    parser.add_argument('--use-lf-cbf-only',
                        action='store_true',
                        default=False,
                        help='Use lane following CBF only')
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
        device = 'cpu'
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
        model.first = 1
        model2 = LitModel.load_from_checkpoint(args.ckpt_sc)
        model3 = LitModel.load_from_checkpoint(args.ckpt_sc3)
        model4 = LitModel.load_from_checkpoint(args.ckpt_sc4)
        model5 = LitModel.load_from_checkpoint(args.ckpt_sc5)
        model6 = LitModel.load_from_checkpoint(args.ckpt_sc6)
        model7 = LitModel.load_from_checkpoint(args.ckpt_sc7)

        model8 = LitModel.load_from_checkpoint(args.ckpt_sc8)
        model9 = LitModel.load_from_checkpoint(args.ckpt_sc9)
        model10 = LitModel.load_from_checkpoint(args.ckpt_sc10)
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
        elif args.model_module == 'bnet_up':             #########used
            model_kwargs = {'solver': 'cvxpy',
                            'store_intermediate_data': False}
            model.hparams.not_use_gt = True
            if args.use_reference_control:
                model.hparams.model_type = 'deri_ref'
        elif args.model_module == 'abnet' or args.model_module == 'abnet-sc':             #########used
            model_kwargs = {'solver': 'cvxpy',
                            'store_intermediate_data': False}
            model.hparams.not_use_gt = True
            model2.hparams.not_use_gt = True
            model3.hparams.not_use_gt = True
            model4.hparams.not_use_gt = True
            model5.hparams.not_use_gt = True
            model6.hparams.not_use_gt = True
            model7.hparams.not_use_gt = True
            model8.hparams.not_use_gt = True
            model9.hparams.not_use_gt = True
            model10.hparams.not_use_gt = True
            if args.use_reference_control:
                model.hparams.model_type = 'deri_ref'
                model2.hparams.model_type = 'deri_ref'
                model3.hparams.model_type = 'deri_ref'
                model4.hparams.model_type = 'deri_ref'
                model5.hparams.model_type = 'deri_ref'
                model6.hparams.model_type = 'deri_ref'
                model7.hparams.model_type = 'deri_ref'
                model8.hparams.model_type = 'deri_ref'
                model9.hparams.model_type = 'deri_ref'
                model10.hparams.model_type = 'deri_ref'
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
            model2.cuda()
            model3.cuda()
            model4.cuda()
            model5.cuda()
            model6.cuda()
            model7.cuda()
            model8.cuda()
            model9.cuda()
            model10.cuda()
        model.eval()
        model2.eval()
        model3.eval()
        model4.eval()
        model5.eval()
        model6.eval()
        model7.eval()
        model8.eval()
        model9.eval()
        model10.eval()

        if args.state_net_model_module:
            LitModelStateNet = importlib.import_module(f'.{args.state_net_model_module}', 'models').LitModel
            model_state_net = LitModelStateNet.load_from_checkpoint(args.state_net_ckpt)
            if not args.no_cuda:
                model_state_net.cuda()
            model_state_net.eval()
            model.hparams.use_indep_state_net = True
            model.hparams.indep_state_net_output = model_state_net.hparams.output_mode
    
    model.hparams.lf_cbf_threshold = args.lf_cbf_threshold
    model.hparams.use_lf_cbf_only = args.use_lf_cbf_only

    model2.hparams.lf_cbf_threshold = args.lf_cbf_threshold
    model2.hparams.use_lf_cbf_only = args.use_lf_cbf_only
    model3.hparams.lf_cbf_threshold = args.lf_cbf_threshold
    model3.hparams.use_lf_cbf_only = args.use_lf_cbf_only

    model4.hparams.lf_cbf_threshold = args.lf_cbf_threshold
    model4.hparams.use_lf_cbf_only = args.use_lf_cbf_only
    model5.hparams.lf_cbf_threshold = args.lf_cbf_threshold
    model5.hparams.use_lf_cbf_only = args.use_lf_cbf_only

    model6.hparams.lf_cbf_threshold = args.lf_cbf_threshold
    model6.hparams.use_lf_cbf_only = args.use_lf_cbf_only
    model7.hparams.lf_cbf_threshold = args.lf_cbf_threshold
    model7.hparams.use_lf_cbf_only = args.use_lf_cbf_only

    model8.hparams.lf_cbf_threshold = args.lf_cbf_threshold
    model8.hparams.use_lf_cbf_only = args.use_lf_cbf_only
    model9.hparams.lf_cbf_threshold = args.lf_cbf_threshold
    model9.hparams.use_lf_cbf_only = args.use_lf_cbf_only
    model10.hparams.lf_cbf_threshold = args.lf_cbf_threshold
    model10.hparams.use_lf_cbf_only = args.use_lf_cbf_only

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

    
    loc_xy, barrier, control, ctime, crash, out_lane, n_passed = [], [], [], [], 0, 0, 0
    import time

    stop_eval = False
    for episode_i in range(args.n_episodes):
        print(f'Episode {episode_i:04d}')
        observations = env.reset()
        if args.use_display:
            display.reset()  # reset should be called after env reset
        ego_agent._ego_dynamics._v = args.ego_init_v
        ego_agent._ego_dynamics._d = -2
        # ego_agent._ego_dynamics._d = 2*(np.random.rand(1)[0] - 0.5)    # random initial lateral distance

        if args.out_dir:
            episode_results = []

        if args.save_video:
            assert args.out_dir is not None
            video_dir = os.path.join(args.out_dir, 'video')
            if not os.path.isdir(video_dir):
                os.makedirs(video_dir)
            video_path = os.path.join(video_dir, f'episode_{episode_i:04d}.mp4')
            rate = f'{(1. / dt)}'
            video_writer = FFmpegWriter(video_path,
                                        inputdict={'-r': rate},
                                        outputdict={'-vcodec': 'libx264',
                                                    '-pix_fmt': 'yuv420p',
                                                    '-r': rate,})

        done = False
        if hasattr(model, 'get_initial_state'):
            rnn_state = model.get_initial_state(batch_size=1)
            rnn_state2 = model.get_initial_state(batch_size=1)
            rnn_state3 = model.get_initial_state(batch_size=1)
            rnn_state4 = model.get_initial_state(batch_size=1)
            rnn_state5 = model.get_initial_state(batch_size=1)
            rnn_state6 = model.get_initial_state(batch_size=1)
            rnn_state7 = model.get_initial_state(batch_size=1)
            rnn_state8 = model.get_initial_state(batch_size=1)
            rnn_state9 = model.get_initial_state(batch_size=1)
            rnn_state10 = model.get_initial_state(batch_size=1)
            if not args.no_cuda:
                if args.model_module == 'abnet' or args.model_module == 'abnet-sc':
                    rnn_state = [[_v.cuda() for _v in rnn_state[head]] for head in range(len(rnn_state))]   # NOTE: change here
                    rnn_state2 = [[_v.cuda() for _v in rnn_state2[head]] for head in range(len(rnn_state2))]   # NOTE: change here
                    rnn_state3 = [[_v.cuda() for _v in rnn_state3[head]] for head in range(len(rnn_state3))]

                    rnn_state4 = [[_v.cuda() for _v in rnn_state4[head]] for head in range(len(rnn_state4))]   # NOTE: change here
                    rnn_state5 = [[_v.cuda() for _v in rnn_state5[head]] for head in range(len(rnn_state5))] 

                    rnn_state6 = [[_v.cuda() for _v in rnn_state6[head]] for head in range(len(rnn_state6))]   # NOTE: change here
                    rnn_state7 = [[_v.cuda() for _v in rnn_state7[head]] for head in range(len(rnn_state7))]

                    rnn_state8 = [[_v.cuda() for _v in rnn_state8[head]] for head in range(len(rnn_state8))] 
                    rnn_state9 = [[_v.cuda() for _v in rnn_state9[head]] for head in range(len(rnn_state9))]   # NOTE: change here
                    rnn_state10 = [[_v.cuda() for _v in rnn_state10[head]] for head in range(len(rnn_state10))]  
                else:
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
        loc_xy_i, barrier_i, control_i, time_i = [], [], [], []
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
                        start = time.time()
                        pred, rnn_state = model(model_inputs, rnn_state, **model_kwargs)
                        model2.p = model.p
                        pred2, rnn_state2 = model2(model_inputs, rnn_state2, **model_kwargs)
                        model3.p = model.p
                        pred3, rnn_state3 = model3(model_inputs, rnn_state3, **model_kwargs)

                        model4.p = model.p
                        pred4, rnn_state4 = model4(model_inputs, rnn_state4, **model_kwargs)
                        model5.p = model.p
                        pred5, rnn_state5 = model5(model_inputs, rnn_state5, **model_kwargs)

                        model6.p = model.p
                        pred6, rnn_state6 = model6(model_inputs, rnn_state6, **model_kwargs)
                        model7.p = model.p
                        pred7, rnn_state7 = model7(model_inputs, rnn_state7, **model_kwargs)

                        model8.p = model.p
                        pred8, rnn_state8 = model8(model_inputs, rnn_state8, **model_kwargs)
                        model9.p = model.p
                        pred9, rnn_state9 = model9(model_inputs, rnn_state9, **model_kwargs)
                        model10.p = model.p
                        pred10, rnn_state10 = model10(model_inputs, rnn_state10, **model_kwargs)
                        wt = 1/10
                        pred = wt*pred + wt*pred2 + wt*pred3 + wt*pred4 + wt*pred5 + wt*pred6 + wt*pred7 + wt*pred8 + wt*pred9 + wt*pred10 # scale
                        termi = time.time()
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
                    img = utils.add_descriptions(env, ego_agent, model, pred, img)

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
                has_crashed = np.any([v['infos'][ego_agent.id]['crashed'] for v in episode_results])

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
            except:
                break
            
            control_i.append([pred[2].item(), pred[3].item()])  #acc, omega
            loc_xy_i.append([env.world.agents[0].ego_dynamics.x, env.world.agents[0].ego_dynamics.y])
            time_i.append(termi - start)
            barrier_i.append(model.barrier.cpu().numpy())

        if has_crashed:
            crash = crash + 1
        if step_results['infos'][ego_agent.id]['out_of_lane']:
            out_lane = out_lane + 1
        if step_results['infos'][ego_agent.id]['n_passed']:
            n_passed = n_passed + 1

        if step_i == 100:
            control.append(control_i)
            loc_xy.append(loc_xy_i)

        time_i = np.array(time_i)
        ctime.append(np.mean(time_i))
        barrier_i = np.array(barrier_i)
        barrier.append(np.min(barrier_i))       

        print(step_results['infos'][ego_agent.id], step_i, 'episode:', episode_i)

        if args.out_dir:
            all_results.append(episode_results)

        if args.save_video:
            video_writer.close()
        if stop_eval:
            break
    
    control = np.array(control)
    loc_xy = np.array(loc_xy)
    ctime = np.array(time_i)
    barrier = np.array(barrier)
    # np.save('./eval_tools/temp_data/abnet_att2_control.npy', control)
    # np.save('./eval_tools/temp_data/abnet_att2_loc.npy', loc_xy)
    # np.save('./eval_tools/temp_data/abnet_att2_time.npy', ctime)
    # np.save('./eval_tools/temp_data/abnet_att2_barrier.npy', barrier)

    # np.save('./eval_tools/temp_data/abnet2_control.npy', control)
    # np.save('./eval_tools/temp_data/abnet2_loc.npy', loc_xy)
    # np.save('./eval_tools/temp_data/abnet2_time.npy', ctime)
    # np.save('./eval_tools/temp_data/abnet2_barrier.npy', barrier)

    # np.save('./eval_tools/temp_data/abnet_sc2_control.npy', control)
    # np.save('./eval_tools/temp_data/abnet_sc2_loc.npy', loc_xy)
    # np.save('./eval_tools/temp_data/abnet_sc2_time.npy', ctime)
    # np.save('./eval_tools/temp_data/abnet_sc2_barrier.npy', barrier)

    # np.save('./eval_tools/temp_data/bnet-up_control.npy', control)
    # np.save('./eval_tools/temp_data/bnet-up_loc.npy', loc_xy)
    # np.save('./eval_tools/temp_data/bnet-up_time.npy', ctime)
    # np.save('./eval_tools/temp_data/bnet-up_barrier.npy', barrier)

    # np.save('./eval_tools/temp_data/bnet2_control.npy', control)
    # np.save('./eval_tools/temp_data/bnet2_loc.npy', loc_xy)
    # np.save('./eval_tools/temp_data/bnet2_time.npy', ctime)
    # np.save('./eval_tools/temp_data/bnet2_barrier.npy', barrier)

    # np.save('./eval_tools/temp_data/dfb3_control.npy', control)
    # np.save('./eval_tools/temp_data/dfb3_loc.npy', loc_xy)
    # np.save('./eval_tools/temp_data/dfb3_time.npy', ctime)
    # np.save('./eval_tools/temp_data/dfb3_barrier.npy', barrier)

    # np.save('./eval_tools/temp_data/e2e3_control.npy', control)
    # np.save('./eval_tools/temp_data/e2e3_loc.npy', loc_xy)
    # np.save('./eval_tools/temp_data/e2e3_time.npy', ctime)
    # np.save('./eval_tools/temp_data/e2e3_barrier.npy', barrier)

    print('crashed:', crash)
    print('out_of_lane:', out_lane)
    print('n_passed:', n_passed)
    print('computation time ave:', np.mean(ctime), 'std:', np.std(ctime))
    print('safety:', np.min(barrier))
    print('conservativeness ave:', np.mean(barrier), 'std:', np.std(barrier))
    print('smooth ave:', np.mean(np.std(control, axis = 0), axis = 0))

    if args.out_dir:
        results_path = os.path.join(args.out_dir, 'results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(all_results, f)


if __name__ == '__main__':
    main()
