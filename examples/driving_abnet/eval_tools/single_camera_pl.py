import collections
import os, sys
from typing import Optional, Tuple
import time
import json
import importlib

import numpy as np
import scipy.interpolate
import cv2
import torch
import torch.backends.cudnn
from PIL import Image
from torch import nn
from torchsparse import SparseTensor

from .. import Callback
from ..utils import functional as F
from ..utils.config import configs
from ..utils.quantize import sparse_quantize
from .base import DeepknightController
from . import preprocess

bnet_root = os.environ.get('BNET_ROOT')
sys.path.insert(0, bnet_root)
from data_modules.utils import transform_rgb

__all__ = ['SingleCameraPLController']

USE_CUDA = True


class SingleCameraPLController(DeepknightController):
    def __init__(self,
                 *,
                 config_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 navigator_path: Optional[str] = None,
                 **unused) -> None:
        super().__init__(navigator_path=navigator_path)
        
        self.master_callback = Callback.IMAGE
        torch.set_grad_enabled(False)
        self.dt = 1 / 30. #0.1
        self.wheel_base = 2.78

        # load configuration
        config_path = model_path # NOTE: [hacky] use model_path to specify config

        configs.reload(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'assets', 'default.yaml'),
                       recursive=False)
        if config_path is None:
            config_path = os.path.join(os.path.split(model_path)[0], 'configs.yaml')
        configs.load(config_path, recursive=False)
        print('Loaded configs from "{}"'.format(config_path))
        print(configs)

        # camera setup
        self.camera.resize(*configs.dataset.camera_size)
        self.roi = self.camera.get_roi()

        # load model
        model_config = configs['model']

        LitModel = importlib.import_module('.{}'.format(model_config['model_module']), 'models').LitModel
        model = LitModel.load_from_checkpoint(configs['model']['ckpt'])
        if model_config['model_module'] == 'barrier_net':
            model_kwargs = {'solver': 'cvxpy',
                            'store_intermediate_data': True}
            model.hparams.not_use_gt = True
            model.hparams.model_type = model_config.get('model_type', model.hparams.model_type)
            model.hparams.use_lane_keeping_CBFs = model_config.get('use_lane_keeping_CBFs', False)
        elif model_config['model_module'] == 'old_barrier_net':
            model_kwargs = {'solver': 'cvxpy',
                            'store_intermediate_data': True}
            model.hparams.not_use_gt = True
            model.hparams.use_color_jitter = False
            model.hparams.use_fixed_standardize = True
        else:
            model_kwargs = dict()
        if USE_CUDA:
            model.cuda()
        model.eval()

        if model_config['state_net_model_module']:
            LitModelStateNet = importlib.import_module('.{}'.format(model_config['state_net_model_module']), 'models').LitModel
            model_state_net = LitModelStateNet.load_from_checkpoint(model_config['state_net_ckpt'])
            if USE_CUDA:
                model_state_net.cuda()
            model_state_net.eval()
            model.hparams.use_indep_state_net = True
            model.hparams.indep_state_net_output = model_state_net.hparams.output_mode
            self.model_state_net = model_state_net

        self.model_kwargs = model_kwargs
        self.model_config = model_config
        self.configs = configs
        self.model = model

    def forward(self,
                measured_curvature,
                speed,
                current_dist: float,
                fcamera: Optional[np.ndarray] = None,
                event_camera: Optional[np.ndarray] = None,
                lidar: Optional[np.ndarray] = None,
                gps: Optional[Tuple[float, float, float]] = None,
                return_full_pred: Optional[bool] = False) -> float:

        # preprocess image
        fcamera = Image.fromarray(np.uint8(fcamera))
        fcamera = F.resize(fcamera, configs.dataset.camera_size)
        (i1, j1, i2, j2) = self.roi
        fcamera = fcamera.crop((j1, i1, j2, i2))
        imgs = np.array(fcamera)
        imgs, _ = transform_rgb(imgs, train=False,
                                use_color_jitter=self.model.hparams.use_color_jitter,
                                use_fixed_standardize=self.model.hparams.use_fixed_standardize)
        imgs = imgs[None, None, ...] # add batch and time dimension

        # prepare model inputs (b,)
        delta = curvature2tireangle(measured_curvature, self.wheel_base)
        v = speed
        s, d, mu, kappa = 0., 0., 0., 0.
        state_data = torch.Tensor([s, d, mu, v, delta, kappa])[None,None,:].to(imgs) # to be inferred
        obs_data = torch.zeros(1, 1, 2).to(imgs) # to be inferred
        ctrl_data = torch.zeros(1, 1, 4).to(imgs) # dummy during eval; not used
        model_inputs = [imgs, state_data, obs_data, ctrl_data]
        if USE_CUDA:
            model_inputs = [v.cuda() for v in model_inputs]

        # predict state
        if self.model_config['state_net_model_module']:
            if not hasattr(self, 'rnn_state_state_net'):
                if hasattr(self.model_state_net, 'get_initial_state'):
                    self.rnn_state_state_net = self.model_state_net.get_initial_state(batch_size=1)
                    if USE_CUDA:
                        if isinstance(self.rnn_state_state_net, list):
                            self.rnn_state_state_net = [v.cuda() for v in self.rnn_state_state_net]
                        elif isinstance(self.rnn_state_state_net, dict):
                            self.rnn_state_state_net = {k: [vv.cuda for vv in v] for k, v in self.rnn_state_state_net.items()}
                        else:
                            raise NotImplementedError
                else:
                    self.rnn_state_state_net = None
            with torch.no_grad():
                pred_state_net, self.rnn_state_state_net = self.model_state_net(model_inputs,
                    self.rnn_state_state_net, **self.model_kwargs)
            if self.model_config.set_obs_d_lower_bound is not None:
                obs_d_idx = self.model_state_net.hparams.output_mode.index('obs_d')
                if torch.sign(pred_state_net[:, :, obs_d_idx]) > 0:
                    pred_state_net[:, :, obs_d_idx] = torch.clamp(
                        pred_state_net[:, :, obs_d_idx], min=self.model_config.set_obs_d_lower_bound)
                else:
                    pred_state_net[:, :, obs_d_idx] = torch.clamp(
                        pred_state_net[:, :, obs_d_idx], max=-self.model_config.set_obs_d_lower_bound)
            if self.model_state_net.hparams.drop_obs_d_offset:
                if 'obs_d' in self.model_state_net.hparams.output_mode:
                    obs_d_idx = self.model_state_net.hparams.output_mode.index('obs_d')
                    print('obs_d', pred_state_net[:, :, obs_d_idx])
                    pred_state_net[:, :, obs_d_idx] += \
                        torch.sign(pred_state_net[:, :, obs_d_idx]) * 6
                if 'dd' in self.model_state_net.hparams.output_mode:
                    dd_idx = self.model_state_net.hparams.output_mode.index('dd')
                    pred_state_net[:, :, dd_idx] += \
                        torch.sign(pred_state_net[:, :, dd_idx]) * 5
            model_inputs.append(pred_state_net)

        # policy take action
        if not hasattr(self, 'rnn_state'):
            if hasattr(self.model, 'get_initial_state'):
                self.rnn_state = self.model.get_initial_state(batch_size=1)
                if isinstance(self.rnn_state, list):
                    self.rnn_state = [v.cuda() for v in self.rnn_state]
                elif isinstance(self.rnn_state, dict):
                    self.rnn_state = {k: [vv.cuda for vv in v] for k, v in self.rnn_state.items()}
                else:
                    raise NotImplementedError
            else:
                self.rnn_state = None
        with torch.no_grad():
            pred, self.rnn_state = self.model(model_inputs, self.rnn_state, **self.model_kwargs)
        pred = pred.cpu().numpy()[0, 0] # drop batch and time dimension

        print(self.model.intermediate_data['ds'], self.model.intermediate_data['dd'])

        if return_full_pred:
            return pred
        else:
            omega = pred[self.model.hparams.output_mode.index('omega')]
            a = pred[self.model.hparams.output_mode.index('a')]
            
            # integrate to curvature
            if self.model_config['integrate_control']:
                new_delta = delta + self.dt * omega
                inverse_r = tireangle2curvature(new_delta, self.wheel_base)

                new_v = v + self.dt * a
                self.speed_control = new_v

                return inverse_r
            else:
                return omega, a

    def draw_gui(self, img, curv_model, curv_true, event_img=None) -> None:
        """ Simple method for displaying a GUI while driving """

        if img is None:
            return

        # show image on screen
        image_resized = cv2.resize(img, (self.camera.get_width(), self.camera.get_height()))
        # self.Image.draw_box_new(image_resized, self.roi)

        noodle_coords_model = self.Image.draw_noodle(curv_model, self.camera)
        noodle_coords_true = self.Image.draw_noodle(curv_true, self.camera)
        cv2.polylines(image_resized, [noodle_coords_model.T], False,
                      (255, 0, 0), 2)
        cv2.polylines(image_resized, [noodle_coords_true.T], False,
                      (0, 0, 255), 2)

        if event_img is not None:
            event_img = cv2.resize(event_img, (self.camera.get_width(), self.camera.get_height()))
            if img is None:
                image_resized = event_img
            else:
                image_resized = np.concatenate([image_resized, event_img], axis=1)

        cv2.imshow('Deepknight', image_resized)
        cv2.waitKey(1)


def curvature2tireangle(curvature: float, wheel_base: float) -> float:
    return np.arctan(wheel_base * curvature)


def tireangle2curvature(tire_angle: float, wheel_base: float) -> float:
    return np.tan(tire_angle) / wheel_base
