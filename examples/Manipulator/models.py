import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from qpth.qp import QPFunction
import numpy as np
from my_classes import test_solver as solver
from eBQP import eBQP


class BarrierNet(nn.Module):
    def __init__(self, model_param, mean, std, mean_label, std_label, device, bn, activation = 'relu'):
        super().__init__()
        self.bn = bn
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.mean_label = torch.from_numpy(mean_label).to(device)
        self.std_label = torch.from_numpy(std_label).to(device)
        self.device = device
        self.obs_x = 0  #obstacle location
        self.obs_y = 7
        self.R = 4   #obstacle size
        self.p1 = 0
        self.p2 = 0
        self.nCls = model_param[7]

        self.abnet = True
        self.x52 = 0

        self.add_noise = False
        
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(model_param[1])
            self.bn2 = nn.BatchNorm1d(model_param[2])
            self.bn31 = nn.BatchNorm1d(model_param[3])

        self.fc1 = nn.Linear(model_param[0], model_param[1]).double()
        self.fc2 = nn.Linear(model_param[1], model_param[2]).double()
        self.fc31 = nn.Linear(model_param[2], model_param[3]).double()
        self.fc32 = nn.Linear(model_param[2], model_param[4]).double()
        self.fc41 = nn.Linear(model_param[3], model_param[5]).double()
        self.fc42 = nn.Linear(model_param[4], model_param[6]).double()
        self.fc51 = nn.Linear(model_param[5], model_param[7]).double()
        self.fc52 = nn.Linear(model_param[6], model_param[7]).double()

        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError(f'Not supported activation function {activation}')

        # QP params.
        # from previous layers

    def forward(self, x, sgn):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x0 = x*self.std + self.mean
        if self.add_noise:
            x1 = self.activation(self.fc1(x + 0.1*torch.randn_like(x)))
        else:
            x1 = self.activation(self.fc1(x))
        if self.bn:
            x1 = self.bn1(x1)
        
        x2 = self.activation(self.fc2(x1))
        if self.bn:
            x2 = self.bn2(x2)

        x31 = self.activation(self.fc31(x2))
        x32 = self.activation(self.fc32(x2))

        x41 = self.activation(self.fc41(x31))
        x42 = self.activation(self.fc42(x32))

        x51 = self.fc51(x41)
        x52 = 4*nn.Sigmoid()(self.fc52(x42))  # ensure CBF parameters are positive

        if self.abnet:
            self.x52 = x52
        
        # BarrierNet
        x = self.dCBF(x0, x51, x52, sgn, nBatch)
               
        return x

    def dCBF(self, x0, x51, x52, sgn, nBatch):
        l1, l2 = 3, 3       #arm link length

        Q = Variable(torch.eye(self.nCls))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)
        theta1 = x0[:,0]
        w1 = x0[:,1]
        theta2 = x0[:,2]
        w2 = x0[:,3]
        sint1 = torch.sin(theta1)
        sint2 = torch.sin(theta2)
        cost1 = torch.cos(theta1)
        cost2 = torch.cos(theta2)

        barrier = (l1*cost1 + l2*cost2 - self.obs_x)**2 + (l1*sint1 + l2*sint2 - self.obs_y)**2 - self.R**2
        b_dot = 2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*sint1*w1 - l2*sint2*w2) + \
            2*(l1*sint1 + l2*sint2 - self.obs_y)*(l1*cost1*w1 + l2*cost2*w2)
        Lf2b = 2*(-l1*sint1*w1 - l2*sint2*w2)**2 + 2*(l1*cost1*w1 + l2*cost2*w2)**2 + \
            2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*cost1*w1**2 - l2*cost2*w2**2) + \
            2*(l1*sint1 + l2*sint2 - self.obs_y)*(-l1*sint1*w1**2 - l2*sint2*w2**2)
        LgLfbu1 = torch.reshape(2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*sint1) + 2*(l1*sint1 + l2*sint2 - self.obs_y)*(l1*cost1), (nBatch, 1))
        LgLfbu2 = torch.reshape(2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l2*sint2) + 2*(l1*sint1 + l2*sint2 - self.obs_y)*(l2*cost2), (nBatch, 1))
        
          
        G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
        G = torch.reshape(G, (nBatch, 1, self.nCls)).to(self.device)
        if self.abnet:     
            h = (torch.reshape(Lf2b + (x52[:,0] + x52[:,1])*b_dot + (x52[:,0]*x52[:,1])*barrier, (nBatch, 1))).to(self.device) 
        else:
            h = (torch.reshape(Lf2b + (self.x52[:,0] + x52[:,1])*b_dot + (self.x52[:,0]*x52[:,1])*barrier, (nBatch, 1))).to(self.device)
        e = Variable(torch.Tensor()).to(self.device)
            
        if self.training or sgn == 1:    
            x = QPFunction(verbose = 0)(Q.double(), x51.double(), G.double(), h.double(), e, e)
            x = (x - self.mean_label)/self.std_label   # normalize output
        else:
            self.p1 = x52[0,0]
            self.p2 = x52[0,1]
            x = solver(Q[0].double(), x51[0].double(), G[0].double(), h[0].double())
            x = torch.tensor([[x[0], x[1]]]).to(self.device)
            # x = x*self.std_label + self.mean_label  # no need to denormalize for testing
        
        return x

class ABNet(nn.Module):
    def __init__(self, model_param, mean, std, mean_label, std_label, device, bn, heads = 10):
        super().__init__()
        self.bn = bn
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.mean_label = torch.from_numpy(mean_label).to(device)
        self.std_label = torch.from_numpy(std_label).to(device)
        self.device = device
        self.obs_x = 0  #obstacle location
        self.obs_y = 7
        self.R = 4   #obstacle size
        self.p1 = 0
        self.p2 = 0
        self.nCls = model_param[7]

        self.x52 = 0
        self.heads = heads

        self.add_noise = False
        self.use_cf = False
        
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(model_param[1])
            self.bn2 = nn.BatchNorm1d(model_param[2])
            self.bn31 = nn.BatchNorm1d(model_param[3])

        self.fc1, self.fc2, self.fc31, self.fc32, self.fc41, self.fc42, self.fc51, self.fc52 = nn.ModuleList(), nn.ModuleList(), \
              nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(self.heads):
            self.fc1.append(nn.Linear(model_param[0], model_param[1]).double().to(device))
            self.fc2.append(nn.Linear(model_param[1], model_param[2]).double().to(device))
            self.fc31.append(nn.Linear(model_param[2], model_param[3]).double().to(device))
            self.fc32.append(nn.Linear(model_param[2], model_param[4]).double().to(device))
            self.fc41.append(nn.Linear(model_param[3], model_param[5]).double().to(device))
            self.fc42.append(nn.Linear(model_param[4], model_param[6]).double().to(device))
            self.fc51.append(nn.Linear(model_param[5], model_param[7]).double().to(device))
            self.fc52.append(nn.Linear(model_param[6], model_param[7]).double().to(device))

        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()
        # QP params.
        # from previous layers
        self.wt = Parameter(torch.ones(self.heads))
    
    def sub_forward(self, x, x0, nBatch, sgn, i): 
        if self.add_noise:    
            x1 = self.activation(self.fc1[i](x + 0.1*torch.randn_like(x)))  #add noise to each model observation
        else:
            x1 = self.activation(self.fc1[i](x))
        if self.bn:
            x1 = self.bn1(x1)
        
        x2 = self.activation(self.fc2[i](x1))
        if self.bn:
            x2 = self.bn2(x2)

        x31 = self.activation(self.fc31[i](x2))
        x32 = self.activation(self.fc32[i](x2))

        x41 = self.activation(self.fc41[i](x31))
        x42 = self.activation(self.fc42[i](x32))

        x51 = self.fc51[i](x41)
        x52 = 4*nn.Sigmoid()(self.fc52[i](x42))  # ensure CBF parameters are positive

        if i == 0:
            self.x52 = x52
        
        # BarrierNet/ABNet
        if self.use_cf:
            u = self.dCBF_cf(x0, x51, x52, sgn, nBatch, i)
        else:
            u = self.dCBF(x0, x51, x52, sgn, nBatch, i)

        return u

    def forward(self, x, sgn, itr = 0):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x0 = x*self.std + self.mean

        wt_sum = 0
        for i in range(self.heads):
            wt_sum = wt_sum + torch.exp(self.wt[i])
        u_sum = 0
        for i in range(self.heads):
            u_sum = u_sum + self.sub_forward(x, x0, nBatch, sgn, i)*torch.exp(self.wt[i])/wt_sum
                     
        return u_sum
    
    def dCBF(self, x0, x51, x52, sgn, nBatch, i):
        l1, l2 = 3, 3       #arm link length

        Q = Variable(torch.eye(self.nCls))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)
        theta1 = x0[:,0]
        w1 = x0[:,1]
        theta2 = x0[:,2]
        w2 = x0[:,3]
        sint1 = torch.sin(theta1)
        sint2 = torch.sin(theta2)
        cost1 = torch.cos(theta1)
        cost2 = torch.cos(theta2)

        barrier = (l1*cost1 + l2*cost2 - self.obs_x)**2 + (l1*sint1 + l2*sint2 - self.obs_y)**2 - self.R**2
        b_dot = 2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*sint1*w1 - l2*sint2*w2) + \
            2*(l1*sint1 + l2*sint2 - self.obs_y)*(l1*cost1*w1 + l2*cost2*w2)
        Lf2b = 2*(-l1*sint1*w1 - l2*sint2*w2)**2 + 2*(l1*cost1*w1 + l2*cost2*w2)**2 + \
            2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*cost1*w1**2 - l2*cost2*w2**2) + \
            2*(l1*sint1 + l2*sint2 - self.obs_y)*(-l1*sint1*w1**2 - l2*sint2*w2**2)
        LgLfbu1 = torch.reshape(2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*sint1) + 2*(l1*sint1 + l2*sint2 - self.obs_y)*(l1*cost1), (nBatch, 1))
        LgLfbu2 = torch.reshape(2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l2*sint2) + 2*(l1*sint1 + l2*sint2 - self.obs_y)*(l2*cost2), (nBatch, 1))
        
          
        G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
        G = torch.reshape(G, (nBatch, 1, self.nCls)).to(self.device)
        if i == 0: # or sgn == 1:     #######################possible problem here with sgn == 1
            h = (torch.reshape(Lf2b + (x52[:,0] + x52[:,1])*b_dot + (x52[:,0]*x52[:,1])*barrier, (nBatch, 1))).to(self.device) 
        else:
            h = (torch.reshape(Lf2b + (self.x52[:,0] + x52[:,1])*b_dot + (self.x52[:,0]*x52[:,1])*barrier, (nBatch, 1))).to(self.device)
        e = Variable(torch.Tensor()).to(self.device)
            
        if self.training or sgn == 1:    
            x = QPFunction(verbose = 0)(Q.double(), x51.double(), G.double(), h.double(), e, e)
            x  = (x - self.mean_label)/self.std_label    # normalize output
        else:
            if i == 0:
                self.p1 = x52[0,0]
                self.p2 = x52[0,1]
            x = solver(Q[0].double(), x51[0].double(), G[0].double(), h[0].double())
            x = torch.tensor([[x[0], x[1]]]).to(self.device)
            # x = x*self.std_label + self.mean_label  # no need to denormalize for testing
        
        return x
    
    def dCBF_cf(self, x0, x51, x52, sgn, nBatch, i):  # closed-form/explicit-Barrier
        l1, l2 = 3, 3       #arm link length

        Q = Variable(torch.eye(self.nCls))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)
        theta1 = x0[:,0]
        w1 = x0[:,1]
        theta2 = x0[:,2]
        w2 = x0[:,3]
        sint1 = torch.sin(theta1)
        sint2 = torch.sin(theta2)
        cost1 = torch.cos(theta1)
        cost2 = torch.cos(theta2)

        barrier = (l1*cost1 + l2*cost2 - self.obs_x)**2 + (l1*sint1 + l2*sint2 - self.obs_y)**2 - self.R**2
        b_dot = 2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*sint1*w1 - l2*sint2*w2) + \
            2*(l1*sint1 + l2*sint2 - self.obs_y)*(l1*cost1*w1 + l2*cost2*w2)
        Lf2b = 2*(-l1*sint1*w1 - l2*sint2*w2)**2 + 2*(l1*cost1*w1 + l2*cost2*w2)**2 + \
            2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*cost1*w1**2 - l2*cost2*w2**2) + \
            2*(l1*sint1 + l2*sint2 - self.obs_y)*(-l1*sint1*w1**2 - l2*sint2*w2**2)
        LgLfbu1 = torch.reshape(2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*sint1) + 2*(l1*sint1 + l2*sint2 - self.obs_y)*(l1*cost1), (nBatch, 1))
        LgLfbu2 = torch.reshape(2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l2*sint2) + 2*(l1*sint1 + l2*sint2 - self.obs_y)*(l2*cost2), (nBatch, 1))
        
          
        G1 = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
        # G = torch.reshape(G, (nBatch, 1, self.nCls)).to(self.device)
        if i == 0: # or sgn == 1:     #######################possible problem here with sgn == 1
            h1 = (torch.reshape(Lf2b + (x52[:,0] + x52[:,1])*b_dot + (x52[:,0]*x52[:,1])*barrier, (nBatch, 1))).to(self.device) 
        else:
            h1 = (torch.reshape(Lf2b + (self.x52[:,0] + x52[:,1])*b_dot + (self.x52[:,0]*x52[:,1])*barrier, (nBatch, 1))).to(self.device)

        # control bound u1 - upper, u1 <= 10 (kind of redundant)
        coe_u1 = torch.ones_like(h1).to(self.device)
        coe_u2 = torch.zeros_like(h1).to(self.device)
        G2 = torch.cat([coe_u1, coe_u2], dim=1)
        h2 = torch.ones_like(h1)*10


        G10 = G1.unsqueeze(1)
        G20 = G2.unsqueeze(1)
        G0 = torch.cat([G10, G20],dim=1)
        h0 = torch.cat([h1, h2],dim=1)

        H = Variable(torch.eye(self.nCls))
        H = H.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)
        
        x = eBQP(H, -x51, G0, h0)

        return x


class ABNet_sc(nn.Module):
    def __init__(self, model_param, mean, std, mean_label, std_label, device, bn, heads = 10):
        super().__init__()
        self.bn = bn
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.mean_label = torch.from_numpy(mean_label).to(device)
        self.std_label = torch.from_numpy(std_label).to(device)
        self.device = device
        self.obs_x = 0  #obstacle location
        self.obs_y = 7
        self.R = 4   #obstacle size
        self.p1 = 0
        self.p2 = 0
        self.nCls = model_param[7]

        self.x52 = 0
        self.heads = heads

        self.first = 0

        self.add_noise = False
        
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(model_param[1])
            self.bn2 = nn.BatchNorm1d(model_param[2])
            self.bn31 = nn.BatchNorm1d(model_param[3])

        self.fc1, self.fc2, self.fc31, self.fc32, self.fc41, self.fc42, self.fc51, self.fc52 = nn.ModuleList(), nn.ModuleList(), \
              nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(self.heads):
            self.fc1.append(nn.Linear(model_param[0], model_param[1]).double().to(device))
            self.fc2.append(nn.Linear(model_param[1], model_param[2]).double().to(device))
            self.fc31.append(nn.Linear(model_param[2], model_param[3]).double().to(device))
            self.fc32.append(nn.Linear(model_param[2], model_param[4]).double().to(device))
            self.fc41.append(nn.Linear(model_param[3], model_param[5]).double().to(device))
            self.fc42.append(nn.Linear(model_param[4], model_param[6]).double().to(device))
            self.fc51.append(nn.Linear(model_param[5], model_param[7]).double().to(device))
            self.fc52.append(nn.Linear(model_param[6], model_param[7]).double().to(device))

        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()
        # QP params.
        # from previous layers
        self.wt = Parameter(torch.ones(self.heads))
    
    def sub_forward(self, x, x0, nBatch, sgn, i): 
        if self.add_noise:    
            x1 = self.activation(self.fc1[i](x + 0.1*torch.randn_like(x)))  #add noise to each model observation
        else:
            x1 = self.activation(self.fc1[i](x))
        if self.bn:
            x1 = self.bn1(x1)
        
        x2 = self.activation(self.fc2[i](x1))
        if self.bn:
            x2 = self.bn2(x2)

        x31 = self.activation(self.fc31[i](x2))
        x32 = self.activation(self.fc32[i](x2))

        x41 = self.activation(self.fc41[i](x31))
        x42 = self.activation(self.fc42[i](x32))

        x51 = self.fc51[i](x41)
        x52 = 4*nn.Sigmoid()(self.fc52[i](x42))  # ensure CBF parameters are positive

        if i == 0 and self.first == 1:
            self.x52 = x52
        
        # BarrierNet
        u = self.dCBF(x0, x51, x52, sgn, nBatch, i)

        return u

    def forward(self, x, sgn, itr = 0):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x0 = x*self.std + self.mean

        wt_sum = 0
        for i in range(self.heads):
            wt_sum = wt_sum + torch.exp(self.wt[i])
        u_sum = 0
        for i in range(self.heads):
            u_sum = u_sum + self.sub_forward(x, x0, nBatch, sgn, i)*torch.exp(self.wt[i])/wt_sum
                     
        return u_sum
    
    def dCBF(self, x0, x51, x52, sgn, nBatch, i):
        l1, l2 = 3, 3       #arm link length

        Q = Variable(torch.eye(self.nCls))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)
        theta1 = x0[:,0]
        w1 = x0[:,1]
        theta2 = x0[:,2]
        w2 = x0[:,3]
        sint1 = torch.sin(theta1)
        sint2 = torch.sin(theta2)
        cost1 = torch.cos(theta1)
        cost2 = torch.cos(theta2)

        barrier = (l1*cost1 + l2*cost2 - self.obs_x)**2 + (l1*sint1 + l2*sint2 - self.obs_y)**2 - self.R**2
        b_dot = 2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*sint1*w1 - l2*sint2*w2) + \
            2*(l1*sint1 + l2*sint2 - self.obs_y)*(l1*cost1*w1 + l2*cost2*w2)
        Lf2b = 2*(-l1*sint1*w1 - l2*sint2*w2)**2 + 2*(l1*cost1*w1 + l2*cost2*w2)**2 + \
            2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*cost1*w1**2 - l2*cost2*w2**2) + \
            2*(l1*sint1 + l2*sint2 - self.obs_y)*(-l1*sint1*w1**2 - l2*sint2*w2**2)
        LgLfbu1 = torch.reshape(2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*sint1) + 2*(l1*sint1 + l2*sint2 - self.obs_y)*(l1*cost1), (nBatch, 1))
        LgLfbu2 = torch.reshape(2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l2*sint2) + 2*(l1*sint1 + l2*sint2 - self.obs_y)*(l2*cost2), (nBatch, 1))
        
          
        G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
        G = torch.reshape(G, (nBatch, 1, self.nCls)).to(self.device)
        if i == 0 and self.first == 1: # or sgn == 1:     #######################possible problem here with sgn == 1
            h = (torch.reshape(Lf2b + (x52[:,0] + x52[:,1])*b_dot + (x52[:,0]*x52[:,1])*barrier, (nBatch, 1))).to(self.device) 
        else:
            h = (torch.reshape(Lf2b + (self.x52[:,0] + x52[:,1])*b_dot + (self.x52[:,0]*x52[:,1])*barrier, (nBatch, 1))).to(self.device)
        e = Variable(torch.Tensor()).to(self.device)
            
        if self.training or sgn == 1:    
            x = QPFunction(verbose = 0)(Q.double(), x51.double(), G.double(), h.double(), e, e)
            x  = (x - self.mean_label)/self.std_label    # normalize output
        else:
            if i == 0:
                self.p1 = x52[0,0]
                self.p2 = x52[0,1]
            x = solver(Q[0].double(), x51[0].double(), G[0].double(), h[0].double())
            x = torch.tensor([[x[0], x[1]]]).to(self.device)
            # x = x*self.std_label + self.mean_label  # no need to denormalize for testing
        
        return x


class DFBNet(nn.Module):
    def __init__(self, model_param, mean, std, mean_label, std_label, device, bn):
        super().__init__()
        self.bn = bn
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.mean_label = torch.from_numpy(mean_label).to(device)
        self.std_label = torch.from_numpy(std_label).to(device)
        self.device = device
        self.obs_x = 0  #obstacle location
        self.obs_y = 7
        self.R = 4   #obstacle size
        self.nCls = model_param[7]

        self.add_noise = False

        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(model_param[1])
            self.bn2 = nn.BatchNorm1d(model_param[2])
            self.bn31 = nn.BatchNorm1d(model_param[3])

        self.fc1 = nn.Linear(model_param[0], model_param[1]).double()
        self.fc2 = nn.Linear(model_param[1], model_param[2]).double()
        self.fc31 = nn.Linear(model_param[2], model_param[3]).double()
        self.fc41 = nn.Linear(model_param[3], model_param[5]).double()
        self.fc51 = nn.Linear(model_param[5], model_param[7]).double()

        # QP params.
        # none 

    def forward(self, x, sgn):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        
        x0 = x*self.std + self.mean
        if self.add_noise:
            x1 = F.relu(self.fc1(x + 0.1*torch.randn_like(x)))
        else:
            x1 = F.relu(self.fc1(x))
        if self.bn:
            x1 = self.bn1(x1)
        
        x2 = F.relu(self.fc2(x1))
        if self.bn:
            x2 = self.bn2(x2)

        x31 = F.relu(self.fc31(x2))

        x41 = F.relu(self.fc41(x31))

        x51 = self.fc51(x41)
        
        #return x31
    
        if self.training or sgn == 1:
            return x51
        else:
            x = self.CBF(x0, x51, nBatch)
            return x
        
    def CBF(self, x0, x51, nBatch):
        l1, l2 = 3, 3       #arm link length

        Q = Variable(torch.eye(self.nCls))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)
        theta1 = x0[:,0]
        w1 = x0[:,1]
        theta2 = x0[:,2]
        w2 = x0[:,3]
        sint1 = torch.sin(theta1)
        sint2 = torch.sin(theta2)
        cost1 = torch.cos(theta1)
        cost2 = torch.cos(theta2)

        # import pdb; pdb.set_trace()

        barrier = (l1*cost1 + l2*cost2 - self.obs_x)**2 + (l1*sint1 + l2*sint2 - self.obs_y)**2 - self.R**2
        b_dot = 2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*sint1*w1 - l2*sint2*w2) + \
            2*(l1*sint1 + l2*sint2 - self.obs_y)*(l1*cost1*w1 + l2*cost2*w2)
        Lf2b = 2*(-l1*sint1*w1 - l2*sint2*w2)**2 + 2*(l1*cost1*w1 + l2*cost2*w2)**2 + \
            2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*cost1*w1**2 - l2*cost2*w2**2) + \
            2*(l1*sint1 + l2*sint2 - self.obs_y)*(-l1*sint1*w1**2 - l2*sint2*w2**2)
        LgLfbu1 = torch.reshape(2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*sint1) + 2*(l1*sint1 + l2*sint2 - self.obs_y)*(l1*cost1), (nBatch, 1))
        LgLfbu2 = torch.reshape(2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l2*sint2) + 2*(l1*sint1 + l2*sint2 - self.obs_y)*(l2*cost2), (nBatch, 1))
        
          
        G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
        G = torch.reshape(G, (nBatch, 1, self.nCls)).to(self.device)
        h = (torch.reshape(Lf2b + (0.5 + 0.5)*b_dot + (0.5*0.5)*barrier, (nBatch, 1))).to(self.device) 
        e = Variable(torch.Tensor()).to(self.device)
        ref = x51[0]*self.std_label + self.mean_label
        x = solver(Q[0].double(), -ref.double(), G[0].double(), h[0].double())
        x = torch.tensor([[x[0], x[1]]]).to(self.device)
        # x = x*self.std_label + self.mean_label  # denormalize for testing
        
        return x


class FCNet(nn.Module):
    def __init__(self, model_param, mean, std, mean_label, std_label, device, bn):
        super().__init__()
        self.bn = bn
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.mean_label = torch.from_numpy(mean_label).to(device)
        self.std_label = torch.from_numpy(std_label).to(device)
        self.device = device
        self.nCls = model_param[7]
        
        self.add_noise = False
        
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(model_param[1])
            self.bn2 = nn.BatchNorm1d(model_param[2])
            self.bn31 = nn.BatchNorm1d(model_param[3])

        self.fc1 = nn.Linear(model_param[0], model_param[1]).double()
        self.fc2 = nn.Linear(model_param[1], model_param[2]).double()
        self.fc31 = nn.Linear(model_param[2], model_param[3]).double()
        self.fc41 = nn.Linear(model_param[3], model_param[5]).double()
        self.fc51 = nn.Linear(model_param[5], model_param[7]).double()


    def forward(self, x, sgn):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        if self.add_noise:
            x = F.relu(self.fc1(x + 0.1*torch.randn_like(x)))
        else:
            x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        
        x2 = F.relu(self.fc2(x))
        if self.bn:
            x2 = self.bn2(x2)

        x31 = F.relu(self.fc31(x2))
        x41 = F.relu(self.fc41(x31))
        x51 = self.fc51(x41)

        if sgn == 0:
            x51 = x51*self.std_label + self.mean_label  # denormalize for testing
        
        return x51
        
        
class ABNet_U(nn.Module):
    def __init__(self, model_param, mean, std, mean_label, std_label, device, bn, heads = 10):
        super().__init__()
        self.bn = bn
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.mean_label = torch.from_numpy(mean_label).to(device)
        self.std_label = torch.from_numpy(std_label).to(device)
        self.device = device
        self.obs_x = 0  #obstacle location
        self.obs_y = 7
        self.R = 4   #obstacle size
        self.p1 = 0
        self.p2 = 0
        self.nCls = model_param[7]

        self.heads = heads

        self.add_noise = False
        
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(model_param[1])
            self.bn2 = nn.BatchNorm1d(model_param[2])
            self.bn31 = nn.BatchNorm1d(model_param[3])

        self.fc1 = nn.Linear(model_param[0], model_param[1]).double().to(device)
        self.fc2 = nn.Linear(model_param[1], model_param[2]).double().to(device)
        self.fc31 = nn.Linear(model_param[2], model_param[3]).double().to(device)
        self.fc32 = nn.Linear(model_param[2], model_param[4]).double().to(device)
        self.fc41 = nn.Linear(model_param[3], model_param[5]).double().to(device)
        self.fc42 = nn.Linear(model_param[4], model_param[6]).double().to(device)
        self.fc51 = nn.Linear(model_param[5], model_param[7]*heads).double().to(device)
        self.fc52 = nn.Linear(model_param[6], heads + 1).double().to(device)

        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()
        # QP params.
        # from previous layers
        self.wt = Parameter(torch.ones(self.heads))
    
    def sub_forward(self, x, x0, nBatch, sgn, i): 
        if self.add_noise:    
            x1 = self.activation(self.fc1(x + 0.1*torch.randn_like(x))) 
        else:
            x1 = self.activation(self.fc1(x))
        if self.bn:
            x1 = self.bn1(x1)
        
        x2 = self.activation(self.fc2(x1))
        if self.bn:
            x2 = self.bn2(x2)

        x31 = self.activation(self.fc31(x2))
        x32 = self.activation(self.fc32(x2))

        x41 = self.activation(self.fc41(x31))
        x42 = self.activation(self.fc42(x32))

        x51 = self.fc51(x41)
        x52 = 4*nn.Sigmoid()(self.fc52(x42))  # ensure CBF parameters are positive
        
        # BarrierNet
        u = self.dCBF(x0, x51, x52, sgn, nBatch, i)

        return u

    def forward(self, x, sgn, itr = 0):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x0 = x*self.std + self.mean
        
       

        u_sum = self.sub_forward(x, x0, nBatch, sgn, 0)
                     
        return u_sum
    
    def dCBF(self, x0, x51, x52, sgn, nBatch, i):
        l1, l2 = 3, 3       #arm link length

        ref = []
        for k in range(self.heads):
            ref.append(x51[:,k*2:k*2+2].unsqueeze(0))
        ref = torch.cat(ref, dim = 0)

        Q = Variable(torch.eye(self.nCls))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls)
        Q = Q.unsqueeze(0).expand(self.heads, nBatch, self.nCls, self.nCls).to(self.device)
        theta1 = x0[:,0]
        w1 = x0[:,1]
        theta2 = x0[:,2]
        w2 = x0[:,3]
        sint1 = torch.sin(theta1)
        sint2 = torch.sin(theta2)
        cost1 = torch.cos(theta1)
        cost2 = torch.cos(theta2)

        barrier = (l1*cost1 + l2*cost2 - self.obs_x)**2 + (l1*sint1 + l2*sint2 - self.obs_y)**2 - self.R**2
        b_dot = 2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*sint1*w1 - l2*sint2*w2) + \
            2*(l1*sint1 + l2*sint2 - self.obs_y)*(l1*cost1*w1 + l2*cost2*w2)
        Lf2b = 2*(-l1*sint1*w1 - l2*sint2*w2)**2 + 2*(l1*cost1*w1 + l2*cost2*w2)**2 + \
            2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*cost1*w1**2 - l2*cost2*w2**2) + \
            2*(l1*sint1 + l2*sint2 - self.obs_y)*(-l1*sint1*w1**2 - l2*sint2*w2**2)
        LgLfbu1 = torch.reshape(2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l1*sint1) + 2*(l1*sint1 + l2*sint2 - self.obs_y)*(l1*cost1), (nBatch, 1))
        LgLfbu2 = torch.reshape(2*(l1*cost1 + l2*cost2 - self.obs_x)*(-l2*sint2) + 2*(l1*sint1 + l2*sint2 - self.obs_y)*(l2*cost2), (nBatch, 1))
        
          
        G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
        G = torch.reshape(G, (nBatch, 1, self.nCls)).to(self.device)
        G = G.unsqueeze(0).expand(self.heads, nBatch, 1, self.nCls).to(self.device)

        h = []
        for k in range(self.heads):
            h0 = (torch.reshape(Lf2b + (x52[:,0] + x52[:,k+1])*b_dot + (x52[:,0]*x52[:,k+1])*barrier, (nBatch, 1))).unsqueeze(0) 
            h.append(h0)
        h = torch.cat(h, dim = 0).to(self.device)

        e = Variable(torch.Tensor()).to(self.device)

        G = torch.reshape(G, (self.heads*nBatch, 1, self.nCls))
        h = torch.reshape(h, (self.heads*nBatch, 1))
        Q = torch.reshape(Q, (self.heads*nBatch, self.nCls, self.nCls))
        ref = torch.reshape(ref, (self.heads*nBatch, self.nCls))
            
        if self.training:
            x = QPFunction(verbose = 0)(Q.double(), ref.double(), G.double(), h.double(), e, e)

            x = torch.reshape(x, (self.heads, nBatch, self.nCls))
            
            wt_vector = torch.exp(self.wt)/torch.sum(torch.exp(self.wt), dim = 0)
            rt = 0
            for k in range(self.heads):
                rt = rt + x[k,:,:]*wt_vector[k]
            # x = torch.sum(wt_vector*x, dim = 0)   #weighted sum

            rt  = (rt - self.mean_label)/self.std_label    # normalize output
        else:
            rt = 0
            wt_vector = torch.exp(self.wt)/torch.sum(torch.exp(self.wt), dim = 0)
            for k in range(self.heads):
                x = solver(Q[k].double(), ref[k].double(), G[k].double(), h[k].double())
                x = torch.tensor([[x[0], x[1]]]).to(self.device)
                rt = rt + x*wt_vector[k]
        
        return rt
        