"""
Example usage
(on Lexus)
>> dk
>> export BNET_ROOT=/home/amini/workspace/driving-BarrierNet
>> export PYOPENGL_PLATFORM=egl # for offscreen rendering
>> export EGL_DEVICE_ID=1 # for offscreen rendering
(change GL_MAX_SAMPLES from 4 to 1 in pyrender/pyrender/renderer.py)
>> MESA_GL_VERSION_OVERRIDE=4.1 python deepknight_on_vista.py # the macro is for offscreen rendering
    --trace-paths ~/workspace/vista/data/20220113-135302_lexus_devens_outerloop
    --mesh-dir ~/workspace/vista/data/carpack01
    --use-curvilinear-dynamics
    --use-display --max-step 100 --save-video ./video.mp4
    --controller-type single_camera_pl --model-path /home/amini/models/rss2022/configs.yaml
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys
import argparse
import inspect
import cv2

import numpy as np
from skvideo.io import FFmpegWriter

import deepknight
import vista

bnet_root = os.environ.get('BNET_ROOT')
sys.path.insert(0, bnet_root)
from eval_tools import utils


## Macros
wheel_base = 2.78
dt = 0.1
sensor_name = 'camera_front'
current_dist = 0.


## Instantiate the deepknight object. The model path will be passed in as
#  a command line arg to ROS when launching the car.
parser = argparse.ArgumentParser(
    description='Test script for the deepknight deployment class.')
# deepknight
parser.add_argument('--controller-type',
                    type=str,
                    default=None,
                    help='type of the controller (see controllers).')
parser.add_argument('--config-path',
                    type=str,
                    default=None,
                    help='path to the config.')
parser.add_argument('--model-path',
                    type=str,
                    default=None,
                    help='path to the model.')
parser.add_argument('--navigator-path',
                    type=str,
                    default=None,
                    help='path to the navigator.')
# vista
parser.add_argument('--trace-paths',
                    type=str,
                    nargs='+',
                    help='Paths to the traces to use for simulation.')
parser.add_argument('--mesh-dir',
                    type=str,
                    default=None,
                    help='Directory of meshes.')
parser.add_argument('--n-agents',
                    type=int,
                    default=2,
                    help='Number of agents')
parser.add_argument('--road-width',
                    type=float,
                    default=4.,
                    help='Road width in VISTA')
parser.add_argument('--reset-mode',
                    type=str,
                    default='uniform',
                    choices=['default', 'segment_start', 'uniform'],
                    help='Trace reset mode in VISTA')
parser.add_argument('--use-display',
                    action='store_true',
                    default=False,
                    help='Use display in vista')
parser.add_argument('--save-video',
                    type=str,
                    default=None,
                    help='Path to save video.')
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
parser.add_argument('--ego-init-v',
                    type=float,
                    default=6.,
                    help='Initial velocity used for acceleration control')
args = parser.parse_args()


## Helper functions
def curvature2tireangle(curvature: float, wheel_base: float) -> float:
    return np.arctan(wheel_base * curvature)


def tireangle2curvature(tire_angle: float, wheel_base: float) -> float:
    return np.tan(tire_angle) / wheel_base


## Spawn the controller and intantiate it
Controller = deepknight.spawn(args.controller_type)
controller = Controller(config_path=args.config_path,
                        model_path=args.model_path,
                        navigator_path=args.navigator_path)
model = controller.model


## Instantiate VISTA
if set(model.hparams.output_mode) == set(['delta', 'v']):
    args.control_mode = 'delta-v'
elif set(model.hparams.output_mode) == set(['omega', 'a']):
    args.control_mode = 'omega-a'
elif set(model.hparams.output_mode[:4]) == set(['delta', 'v', 'omega', 'a']):
    args.control_mode = 'omega-a'
else:
    raise NotImplementedError(f'No corresponding control mode for {model.hparams.output_mode}')
env = utils.get_env(args)
if args.use_display:
    display_config = dict(road_buffer_size=1000, )
    display = vista.Display(env.world, display_config=display_config)

    if args.save_video:
        video_path = args.save_video
        rate = f'{(1. / dt)}'
        video_writer = FFmpegWriter(video_path,
                                    inputdict={'-r': rate},
                                    outputdict={'-vcodec': 'libx264',
                                                '-pix_fmt': 'yuv420p',
                                                '-r': rate,})

## Run
done = False
observations = env.reset()
if args.use_display:
    display.reset()
ego_agent = env.world.agents[0]
ego_agent._ego_dynamics._v = args.ego_init_v
step_i = 0
while not done:
    try:
        measured_curvature = tireangle2curvature(ego_agent.ego_dynamics.steering, wheel_base)
        speed = ego_agent.ego_dynamics.speed
        fcamera = observations[ego_agent.id][sensor_name]

        pred = controller.forward(measured_curvature, speed, current_dist,
                                  fcamera=fcamera, return_full_pred=True)
        actions = utils.construct_actions(env, ego_agent, model, pred)

        observations, rewards, dones, infos = env.step(actions, dt)
        done = dones[ego_agent.id]

        if args.use_display:
            img = display.render()
            img = utils.add_descriptions(env, ego_agent, model, pred, img)

            if args.save_video:
                video_writer.writeFrame(img)

        step_i += 1
        print(step_i)
        if args.max_step:
            done = False
            if step_i >= args.max_step:
                break

    except KeyboardInterrupt:
        if args.save_video:
            video_writer.close()
        break

if args.save_video:
    video_writer.close()
