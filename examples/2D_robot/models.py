import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from qpth.qp import QPFunction
import numpy as np
from my_classes import test_solver as solver
import time
from eBQP import eBQP, eBQP_I, eBQP_g


class BarrierNet(nn.Module):
    def __init__(self, nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.bn = bn
        self.nCls = nCls
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.device = device
        self.obs_x = 40  #obstacle location
        self.obs_y = 15
        self.R = 6   #obstacle size
        self.p1 = 0
        self.p2 = 0

        self.abnet = True
        self.x32 = 0

        self.add_noise = False
        
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden1)
            self.bn21 = nn.BatchNorm1d(nHidden21)
            self.bn22 = nn.BatchNorm1d(nHidden22)

        self.fc1 = nn.Linear(nFeatures, nHidden1).double()
        self.fc21 = nn.Linear(nHidden1, nHidden21).double()
        self.fc22 = nn.Linear(nHidden1, nHidden22).double()
        self.fc31 = nn.Linear(nHidden21, nCls).double()
        self.fc32 = nn.Linear(nHidden22, nCls).double()

        # QP params.
        # from previous layers

    def forward(self, x, sgn):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x0 = x*self.std + self.mean
        if self.add_noise:
            x = F.relu(self.fc1(x + 0.1*torch.randn_like(x)))
        else:
            x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        
        x21 = F.relu(self.fc21(x))
        if self.bn:
            x21 = self.bn21(x21)
        x22 = F.relu(self.fc22(x))
        if self.bn:
            x22 = self.bn22(x22)
        
        x31 = self.fc31(x21)
        x32 = self.fc32(x22)
        x32 = 4*nn.Sigmoid()(x32)  # ensure CBF parameters are positive

        if self.abnet:
            self.x32 = x32
        
        # BarrierNet
        x = self.dCBF(x0, x31, x32, sgn, nBatch)
               
        return x #np.array([-x31[0,0].item(), -x31[0,1].item()])

    def dCBF(self, x0, x31, x32, sgn, nBatch):

        Q = Variable(torch.eye(self.nCls))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)
        px = x0[:,0]
        py = x0[:,1]
        theta = x0[:,2]
        v = x0[:,3]
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        barrier = (px - self.obs_x)**2 + (py - self.obs_y)**2 - self.R**2
        barrier_dot = 2*(px - self.obs_x)*v*cos_theta + 2*(py - self.obs_y)*v*sin_theta
        Lf2b = 2*v**2
        LgLfbu1 = torch.reshape(-2*(px - self.obs_x)*v*sin_theta + 2*(py - self.obs_y)*v*cos_theta, (nBatch, 1)) 
        LgLfbu2 = torch.reshape(2*(px - self.obs_x)*cos_theta + 2*(py - self.obs_y)*sin_theta, (nBatch, 1))
          
        G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
        G = torch.reshape(G, (nBatch, 1, self.nCls)).to(self.device)
        if self.abnet:     
            h = (torch.reshape(Lf2b + (x32[:,0] + x32[:,1])*barrier_dot + (x32[:,0]*x32[:,1])*barrier, (nBatch, 1))).to(self.device) 
        else:
            h = (torch.reshape(Lf2b + (self.x32[:,0] + x32[:,1])*barrier_dot + (self.x32[:,0]*x32[:,1])*barrier, (nBatch, 1))).to(self.device)
        e = Variable(torch.Tensor()).to(self.device)
            
        if self.training or sgn == 1:    
            x = QPFunction(verbose = 0)(Q.double(), x31.double(), G.double(), h.double(), e, e)
        else:
            self.p1 = x32[0,0]
            self.p2 = x32[0,1]
            x = solver(Q[0].double(), x31[0].double(), G[0].double(), h[0].double())
        
        return x

class ABNet(nn.Module):
    def __init__(self, nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn, heads = 10):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.bn = bn
        self.nCls = nCls
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.device = device
        self.obs_x = 40  #obstacle location
        self.obs_y = 15
        self.R = 6   #obstacle size
        self.p1 = 0
        self.p2 = 0
        self.x32 = 0
        self.heads = heads

        self.add_noise = False
        self.use_cf = False  # closed-form solution 

        self.time = []
        self.time_cf = []
        
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden1)
            self.bn21 = nn.BatchNorm1d(nHidden21)
            self.bn22 = nn.BatchNorm1d(nHidden22)

        self.fc1, self.fc21, self.fc22, self.fc31, self.fc32 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(self.heads):
            self.fc1.append(nn.Linear(nFeatures, nHidden1).double().to(device))
            self.fc21.append(nn.Linear(nHidden1, nHidden21).double().to(device))
            self.fc22.append(nn.Linear(nHidden1, nHidden22).double().to(device))
            self.fc31.append(nn.Linear(nHidden21, nCls).double().to(device))
            self.fc32.append(nn.Linear(nHidden22, nCls).double().to(device))

        # QP params.
        # from previous layers
        self.wt = Parameter(torch.ones(self.heads))
    
    def sub_forward(self, x, x0, nBatch, sgn, i): 
        if self.add_noise:    
            x = F.relu(self.fc1[i](x + 0.1*torch.randn_like(x))) 
        else:
            x = F.relu(self.fc1[i](x))
        if self.bn:
            x = self.bn1(x)
        
        x21 = F.relu(self.fc21[i](x))
        if self.bn:
            x21 = self.bn21(x21)
        x22 = F.relu(self.fc22[i](x))
        if self.bn:
            x22 = self.bn22(x22)
        
        x31 = self.fc31[i](x21)
        x32 = self.fc32[i](x22)
        x32 = 4*nn.Sigmoid()(x32)  # ensure CBF parameters are positive
        if i == 0:
            self.x32 = x32
        
        # BarrierNet
        
        if self.use_cf:
            u = self.dCBF_cf(x0, x31, x32, sgn, nBatch, i)
        else:
            u = self.dCBF(x0, x31, x32, sgn, nBatch, i)

        return u

    def forward(self, x, sgn, itr):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x0 = x*self.std + self.mean
                  
        wt_sum = torch.sum(torch.exp(self.wt), dim = 0)
        u_sum = 0
        for i in range(self.heads):
            u_sum = u_sum + self.sub_forward(x, x0, nBatch, sgn, i)*torch.exp(self.wt[i])/wt_sum
 
        return u_sum

    def dCBF(self, x0, x31, x32, sgn, nBatch, i):
        start_time = time.time()
        Q = Variable(torch.eye(self.nCls))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)
        px = x0[:,0]
        py = x0[:,1]
        theta = x0[:,2]
        v = x0[:,3]
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        barrier = (px - self.obs_x)**2 + (py - self.obs_y)**2 - self.R**2
        barrier_dot = 2*(px - self.obs_x)*v*cos_theta + 2*(py - self.obs_y)*v*sin_theta
        Lf2b = 2*v**2
        LgLfbu1 = torch.reshape(-2*(px - self.obs_x)*v*sin_theta + 2*(py - self.obs_y)*v*cos_theta, (nBatch, 1)) 
        LgLfbu2 = torch.reshape(2*(px - self.obs_x)*cos_theta + 2*(py - self.obs_y)*sin_theta, (nBatch, 1))
          
        G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
        G = torch.reshape(G, (nBatch, 1, self.nCls)).to(self.device)  
        if i == 0 or sgn == 1:   
            h = (torch.reshape(Lf2b + (x32[:,0] + x32[:,1])*barrier_dot + (x32[:,0]*x32[:,1])*barrier, (nBatch, 1))).to(self.device)
        else:
            h = (torch.reshape(Lf2b + (self.x32[:,0] + x32[:,0])*barrier_dot + (self.x32[:,0]*x32[:,0])*barrier, (nBatch, 1))).to(self.device) 
        e = Variable(torch.Tensor()).to(self.device)
            
        if self.training or sgn == 1:    
            x = QPFunction(verbose = 0)(Q.double(), -x31.double(), G.double(), h.double(), e, e)  # note -x31 to be consistent with cf
        else:
            if i == 0:
                self.p1 = x32[0,0]
                self.p2 = x32[0,1]
            x = solver(Q[0].double(), -x31[0].double(), G[0].double(), h[0].double())  # note -x31 to be consistent with cf
            x = torch.tensor([[x[0], x[1]]]).to(self.device)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # self.time.append(elapsed_time)
        # print(f"Elapsed time: {elapsed_time} second")
        return x
    
    def dCBF_cf(self, x0, x31, x32, sgn, nBatch, i):  # closed-form solution/explicit-Barrier
        start_time = time.time()
        px = x0[:,0:1]
        py = x0[:,1:2]
        theta = x0[:,2:3]
        v = x0[:,3:4]
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        # obs 1
        barrier = (px - self.obs_x)**2 + (py - self.obs_y)**2 - self.R**2
        barrier_dot = 2*(px - self.obs_x)*v*cos_theta + 2*(py - self.obs_y)*v*sin_theta
        Lf2b = 2*v**2
        LgLfbu1 = -2*(px - self.obs_x)*v*sin_theta + 2*(py - self.obs_y)*v*cos_theta
        LgLfbu2 = 2*(px - self.obs_x)*cos_theta + 2*(py - self.obs_y)*sin_theta
          
        G1 = torch.cat([-LgLfbu1, -LgLfbu2], dim=1).to(self.device) 
        if i == 0 or sgn == 1:   
            h1 = Lf2b + (x32[:,0:1] + x32[:,1:2])*barrier_dot + (x32[:,0:1]*x32[:,1:2])*barrier
        else:
            h1 = Lf2b + (self.x32[:,0:1] + x32[:,0:1])*barrier_dot + (self.x32[:,0:1]*x32[:,0:1])*barrier  # the second should be 1:2, but it is ok due to the symmetric property
        
        # control bound u1 - upper, u1 <= 10 (kind of redundant)
        coe_u1 = torch.ones_like(LgLfbu1).to(self.device)
        coe_u2 = torch.zeros_like(LgLfbu1).to(self.device)
        G2 = torch.cat([coe_u1, coe_u2], dim=1)
        h2 = torch.ones_like(h1)*10

        G10 = G1.unsqueeze(1)
        G20 = G2.unsqueeze(1)
        G0 = torch.cat([G10, G20],dim=1)
        h0 = torch.cat([h1, h2],dim=1)

        H = Variable(torch.eye(self.nCls,dtype=torch.float64))
        H = H.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)
        
        x = eBQP(H, -x31, G0, h0)

        end_time = time.time()
        elapsed_time = end_time - start_time
        # self.time_cf.append(elapsed_time)
        # print(f"Elapsed time: {elapsed_time} second")

        return x
    
    
    

class ABNet_sc(nn.Module):
    def __init__(self, nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn, heads = 10):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.bn = bn
        self.nCls = nCls
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.device = device
        self.obs_x = 40  #obstacle location
        self.obs_y = 15
        self.R = 6   #obstacle size
        self.p1 = 0
        self.p2 = 0
        self.x32 = 0
        self.heads = heads

        self.first = 0

        self.add_noise = False
        
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden1)
            self.bn21 = nn.BatchNorm1d(nHidden21)
            self.bn22 = nn.BatchNorm1d(nHidden22)

        self.fc1, self.fc21, self.fc22, self.fc31, self.fc32 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(self.heads):
            self.fc1.append(nn.Linear(nFeatures, nHidden1).double().to(device))
            self.fc21.append(nn.Linear(nHidden1, nHidden21).double().to(device))
            self.fc22.append(nn.Linear(nHidden1, nHidden22).double().to(device))
            self.fc31.append(nn.Linear(nHidden21, nCls).double().to(device))
            self.fc32.append(nn.Linear(nHidden22, nCls).double().to(device))

        # QP params.
        # from previous layers
        self.wt = Parameter(torch.ones(self.heads))
    
    def sub_forward(self, x, x0, nBatch, sgn, i): 
        if self.add_noise:    
            x = F.relu(self.fc1[i](x + 0.1*torch.randn_like(x))) 
        else:
            x = F.relu(self.fc1[i](x))
        if self.bn:
            x = self.bn1(x)
        
        x21 = F.relu(self.fc21[i](x))
        if self.bn:
            x21 = self.bn21(x21)
        x22 = F.relu(self.fc22[i](x))
        if self.bn:
            x22 = self.bn22(x22)
        
        x31 = self.fc31[i](x21)
        x32 = self.fc32[i](x22)
        x32 = 4*nn.Sigmoid()(x32)  # ensure CBF parameters are positive
        if i == 0 and self.first == 1:
            self.x32 = x32
        
        # BarrierNet
        u = self.dCBF(x0, x31, x32, sgn, nBatch, i)

        return u

    def forward(self, x, sgn, itr):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x0 = x*self.std + self.mean
        
        if 0:#self.training or sgn == 1:
            if itr <= 39:
                u_sum = self.sub_forward(x, x0, nBatch, sgn, np.floor(itr/4).astype(np.int64))
            else:
                # wt_sum = 0
                # for i in range(self.heads):
                #     wt_sum = wt_sum + torch.exp(self.wt[i])
                u_sum = 0
                for i in range(self.heads):
                    u_sum = u_sum + self.sub_forward(x, x0, nBatch, sgn, i)#*torch.exp(self.wt[i])/wt_sum
        else:
            wt_sum = 0
            for i in range(self.heads):
                wt_sum = wt_sum + torch.exp(self.wt[i])
            u_sum = 0
            for i in range(self.heads):
                u_sum = u_sum + self.sub_forward(x, x0, nBatch, sgn, i)*torch.exp(self.wt[i])/wt_sum
                     
        return u_sum

    def dCBF(self, x0, x31, x32, sgn, nBatch, i):

        Q = Variable(torch.eye(self.nCls))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)
        px = x0[:,0]
        py = x0[:,1]
        theta = x0[:,2]
        v = x0[:,3]
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        barrier = (px - self.obs_x)**2 + (py - self.obs_y)**2 - self.R**2
        barrier_dot = 2*(px - self.obs_x)*v*cos_theta + 2*(py - self.obs_y)*v*sin_theta
        Lf2b = 2*v**2
        LgLfbu1 = torch.reshape(-2*(px - self.obs_x)*v*sin_theta + 2*(py - self.obs_y)*v*cos_theta, (nBatch, 1)) 
        LgLfbu2 = torch.reshape(2*(px - self.obs_x)*cos_theta + 2*(py - self.obs_y)*sin_theta, (nBatch, 1))
          
        G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
        G = torch.reshape(G, (nBatch, 1, self.nCls)).to(self.device)  
        if i == 0 and self.first == 1:   # or sgn == 1 for training and valid
            h = (torch.reshape(Lf2b + (x32[:,0] + x32[:,1])*barrier_dot + (x32[:,0]*x32[:,1])*barrier, (nBatch, 1))).to(self.device)
        else:
            h = (torch.reshape(Lf2b + (self.x32[:,0] + x32[:,0])*barrier_dot + (self.x32[:,0]*x32[:,0])*barrier, (nBatch, 1))).to(self.device) 
        e = Variable(torch.Tensor()).to(self.device)
            
        if self.training or sgn == 1:    
            x = QPFunction(verbose = 0)(Q.double(), x31.double(), G.double(), h.double(), e, e)
        else:
            if i == 0:
                self.p1 = x32[0,0]
                self.p2 = x32[0,1]
            x = solver(Q[0].double(), x31[0].double(), G[0].double(), h[0].double())
            x = torch.tensor([[x[0], x[1]]]).to(self.device)
        return x

class DFBNet(nn.Module):
    def __init__(self, nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.nCls = nCls
        self.mean = mean
        self.std = std
        self.device = device
        self.bn = bn
        self.obs_x = 40  #obstacle location
        self.obs_y = 15
        self.R = 6  #obstacle size
        
        self.add_noise = False
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden1)
            self.bn21 = nn.BatchNorm1d(nHidden21)
            self.bn22 = nn.BatchNorm1d(nHidden22)

        self.fc1 = nn.Linear(nFeatures, nHidden1).double()
        self.fc21 = nn.Linear(nHidden1, nHidden21).double()
        self.fc22 = nn.Linear(nHidden1, nHidden22).double()
        self.fc31 = nn.Linear(nHidden21, nCls).double()
        self.fc32 = nn.Linear(nHidden22, nCls).double()

        # QP params.
        # none 

    def forward(self, x, sgn):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        
        x0 = x.cpu()*self.std + self.mean
        if self.add_noise:
            x = F.relu(self.fc1(x + 0.1*torch.randn_like(x)))
        else:
            x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        
        x21 = F.relu(self.fc21(x))
        if self.bn:
            x21 = self.bn21(x21)
        x22 = F.relu(self.fc22(x))
        if self.bn:
            x22 = self.bn22(x22)
        
        x31 = self.fc31(x21)
        
        #return x31
    
        if self.training:
            return x31
        else:
            Q = Variable(torch.eye(self.nCls))
            Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)
            px = x0[:,0]
            py = x0[:,1]
            theta = x0[:,2]
            v = x0[:,3]
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            barrier = (px - self.obs_x)**2 + (py - self.obs_y)**2 - self.R**2
            barrier_dot = 2*(px - self.obs_x)*v*cos_theta + 2*(py - self.obs_y)*v*sin_theta
            Lf2b = 2*v**2
            LgLfbu1 = torch.reshape(-2*(px - self.obs_x)*v*sin_theta + 2*(py - self.obs_y)*v*cos_theta, (nBatch, 1)) 
            LgLfbu2 = torch.reshape(2*(px - self.obs_x)*cos_theta + 2*(py - self.obs_y)*sin_theta, (nBatch, 1))

            G = Variable(torch.tensor(np.append(-LgLfbu1,-LgLfbu2, axis = 1))).to(self.device)      
            h = (torch.reshape(Lf2b, (nBatch, 1)) + torch.reshape((0.5 + 1)*barrier_dot, (nBatch, 1)) + torch.reshape(0.5*barrier, (nBatch, 1))).to(self.device) 

            e = Variable(torch.Tensor()).to(self.device) 

            
            x = solver(Q[0].double(), -x31[0].double(), G.double(), h[0].double())

            return x

class FCNet(nn.Module):
    def __init__(self, nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.nCls = nCls
        self.mean = mean
        self.std = std
        self.device = device
        self.bn = bn
        
        self.add_noise = False
        
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden1)
            self.bn21 = nn.BatchNorm1d(nHidden21)

        self.fc1 = nn.Linear(nFeatures, nHidden1).double()
        self.fc21 = nn.Linear(nHidden1, nHidden21).double()
        self.fc31 = nn.Linear(nHidden21, nCls).double()


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
        
        x21 = F.relu(self.fc21(x))
        if self.bn:
            x21 = self.bn21(x21)
        
        x31 = self.fc31(x21)
        
        return x31
        
        
        