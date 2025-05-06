import torch
import torch.nn as nn
import scipy.io as sio
import models
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
plt.style.use('bmh')
from scipy.integrate import odeint
import os
#export PYTHONNOUSERSITE=True   for python
#export PYOPENGL_PLATFORM=egl   for Vista (cannot connect to "%s"' % name)

#choose abnet or not
abnet = 1
exp = 0


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

#dynamics
def dynamics(y,t):
    dxdt = y[3]*np.cos(y[2])
    dydt = y[3]*np.sin(y[2])
    dttdt = y[4] #u1
    dvdt = y[5]  #u2
    return [dxdt,dydt,dttdt,dvdt, 0, 0]

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using {} device".format(device))
torch.backends.cudnn.benchmark = True

train_data = sio.loadmat('data/dataM_train.mat') #data_train
train_data = train_data['data']
valid_data = sio.loadmat('data/dataM_valid.mat') 
valid_data = valid_data['data']
test_data = sio.loadmat('data/dataM_testn.mat')  # a new destination
test_data = test_data['data']

train0 = np.double(train_data[:,0:5])  # px, py, theta, v, dst
train_labels = np.reshape(np.double(train_data[:,5:7]), (len(train_data),2)) #theta_derivative, acc  4:6, 2
valid0 = np.double(valid_data[:,0:5]) 
valid_labels = np.reshape(np.double(valid_data[:,5:7]), (len(valid_data),2))
test0 = np.double(test_data[:,0:5]) 
test_labels = np.reshape(np.double(test_data[:,5:7]), (len(test_data),2))
init = test0[0]

mean = np.mean(train0, axis = 0)
std= np.std(train0, axis = 0)

train0 = (train0 - mean)/std
valid0 = (valid0 - mean)/std


# Initialize the model.
nFeatures, nHidden1, nHidden21, nHidden22, nCls = 5, 128, 32, 32, 2 
model_list = []
if abnet == 1:
    ## use pre-trained BarrierNet to compose ABNet
    for i in range(10):
        model = models.BarrierNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
        model.load_state_dict(torch.load("./log/bnet{:02d}".format(i+1) + "/model_bn.pth"))
        model.eval() 
        model_list.append(model)
else:
    model = models.FCNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
    model.load_state_dict(torch.load("model_fc.pth"))
    model.eval() 
   
loss_fn = nn.MSELoss()
def test(model):
    px, py, theta, speed, dsty = init[0], init[1], init[2], init[3], init[4]
    tr, tr0 = [], 0
    ctrl1, ctrl2, safety, ctrl1_real, ctrl2_real, p1, p2, loc_x, loc_y = [], [], [], [], [], [], [], [], []
    dt = [0,0.1]
    obs_x, obs_y, R = 40, 15, 6
    true_x, true_y = [], []

    test1 = []

    # running on a vehicle
    with torch.no_grad():
        for i in range(0,len(test0),1): #train0, 10
            #normalize
            x = (px - mean[0])/std[0]
            y = (py - mean[1])/std[1]
            tt = (theta - mean[2])/std[2]
            v = (speed - mean[3])/std[3]
            dst = (dsty - mean[4])/std[4]
            
            #get safety metric
            safe = (px - obs_x)**2 + (py - obs_y)**2 - R**2
            safety.append(safe)
            loc_x.append(px)
            loc_y.append(py)
            true_x.append(test0[i][0])
            true_y.append(test0[i][1])

            #prepare for model input
            x_r = Variable(torch.from_numpy(np.array([x,y,tt,v,dst])), requires_grad=False)
            x_r = torch.reshape(x_r, (1,nFeatures))
            x_r = x_r.to(device)
            ctrl = model(x_r, 0)
            
            test1.append(torch.tensor([[ctrl[0], ctrl[1]]]))
            
            #get the penalty functions
            if abnet == 1:
                p1.append(model.p1.item())  #only for the barriernet
                p2.append(model.p2.item())
            
            #update state
            state = [px,py,theta,speed]
            if abnet == 1:
                state.append(ctrl[0])
                state.append(ctrl[1])
            else:
                state.append(ctrl[0,0].item()) 
                state.append(ctrl[0,1].item())
            
            #update dynamics
            rt = np.float32(odeint(dynamics,state,dt))
            px, py, theta, speed = rt[1][0], rt[1][1], rt[1][2], rt[1][3]
            
            if abnet == 1:
                ctrl1.append(ctrl[0])
                ctrl2.append(ctrl[1])
            else:
                ctrl1.append(ctrl[0,0].item()) 
                ctrl2.append(ctrl[0,1].item())
            ctrl1_real.append(test_labels[i][0])
            ctrl2_real.append(test_labels[i][1])
            tr.append(tr0)
            tr0 = tr0 + dt[1]

        test1 = torch.cat(test1, dim = 0)
        test1_gt = torch.from_numpy(test_labels)
        
        loss = loss_fn(test1, test1_gt)

    return loss.cpu().item()


def abnet_test(model_list, wt_list):
    px, py, theta, speed, dsty = init[0], init[1], init[2], init[3], init[4]
    tr, tr0 = [], 0
    ctrl1, ctrl2, safety, ctrl1_real, ctrl2_real, p1, p2, loc_x, loc_y = [], [], [], [], [], [], [], [], []
    dt = [0,0.1]
    obs_x, obs_y, R = 40, 15, 6
    true_x, true_y = [], []

    # running on a vehicle
    with torch.no_grad():
        for i in range(0,len(test0),1): #train0, 10
            #normalize
            x = (px - mean[0])/std[0]
            y = (py - mean[1])/std[1]
            tt = (theta - mean[2])/std[2]
            v = (speed - mean[3])/std[3]
            dst = (dsty - mean[4])/std[4]
            
            #get safety metric
            safe = (px - obs_x)**2 + (py - obs_y)**2 - R**2
            safety.append(safe)
            loc_x.append(px)
            loc_y.append(py)
            true_x.append(test0[i][0])
            true_y.append(test0[i][1])

            #prepare for model input
            x_r = Variable(torch.from_numpy(np.array([x,y,tt,v,dst])), requires_grad=False)
            x_r = torch.reshape(x_r, (1,nFeatures))
            x_r = x_r.to(device)
            ctrl = 0
            for k in range(10):
                if k == 0:
                    model_list[k].abnet = True
                else:
                    model_list[k].abnet = False
                    model_list[k].x32 = model_list[0].x32
                ctrl = ctrl + model_list[k](x_r, 0)*wt_list[k]
            
            #get the penalty functions
            if abnet == 1:
                p1.append(model.p1.item())  #only for the barriernet
                p2.append(model.p2.item())
            
            #update state
            state = [px,py,theta,speed]
            if abnet == 1:
                state.append(ctrl[0])
                state.append(ctrl[1])
            else:
                state.append(ctrl[0,0].item()) 
                state.append(ctrl[0,1].item())
            
            #update dynamics
            rt = np.float32(odeint(dynamics,state,dt))
            px, py, theta, speed = rt[1][0], rt[1][1], rt[1][2], rt[1][3]
            
            if abnet == 1:
                ctrl1.append(ctrl[0])
                ctrl2.append(ctrl[1])
            else:
                ctrl1.append(ctrl[0,0].item()) 
                ctrl2.append(ctrl[0,1].item())
            ctrl1_real.append(test_labels[i][0])
            ctrl2_real.append(test_labels[i][1])
            tr.append(tr0)
            tr0 = tr0 + dt[1]

    return ctrl1, ctrl2, safety, ctrl1_real, ctrl2_real, p1, p2, loc_x, loc_y, tr, true_x, true_y

# test loss of each BarrierNet, and use the loss to determine the composition weight
loss_list = []
for i in range(10):
    loss = test(model_list[i])
    # if i == 0:
    #     loss = loss/10
    loss_list.append(loss)
wt_list = []
wt_sum = 0
for i in range(10):
    wt_sum = wt_sum + 1/loss_list[i]
for i in range(10):
    wt_list.append(1/loss_list[i]/wt_sum)

# test abnet
ctrl1, ctrl2, safety, ctrl1_real, ctrl2_real, p1, p2, loc_x, loc_y, tr, true_x, true_y = abnet_test(model_list, wt_list)


print("Implementation done!")

makedirs('./log/bnet{:02d}'.format(exp))

plt.figure(1)
fig, ax = plt.subplots()
plt.plot(tr, ctrl1_real, color = 'red', label = 'Ground truth')
plt.plot(tr, ctrl1, color = 'blue', label = 'BarrierNet, R = 6m')
# plt.plot(tr, ctrl1_6_dfb, linestyle='--', color = 'blue', label = 'DFB, R = 6m')
# plt.plot(tr, ctrl1_6_fc, linestyle=':', color = 'blue', label = 'FC, R = 6m')
plt.legend(loc ='upper left', prop={'size': 10})
plt.ylabel('Steering $u_1/(rad/s)$',fontsize=14)
plt.xlabel('time$/s$',fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.set_rasterized(True)
plt.savefig('./log/bnet{:02d}/test_u1'.format(exp))   
# plt.show()

plt.figure(2)
fig, ax = plt.subplots()
plt.plot(tr, ctrl2_real, color = 'red', label = 'Ground truth')
plt.plot(tr, ctrl2, color = 'blue', label = 'BarrierNet, R = 6m')
# plt.plot(tr, ctrl2_6_dfb, linestyle='--', color = 'blue', label = 'DFB, R = 6m')
# plt.plot(tr, ctrl2_6_fc, linestyle=':', color = 'blue', label = 'FC, R = 6m')
plt.legend(prop={'size': 12})
plt.ylabel('Acceleration $u_2/(m/s^2)$',fontsize=14)
plt.xlabel('time$/s$',fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.set_rasterized(True)
plt.savefig('./log/bnet{:02d}/test_u2'.format(exp))
# plt.show()


tt = np.linspace(0, 2*3.14, 60)
r = 6
xx1 = r*np.cos(tt) + 40
yy1 = r*np.sin(tt) + 15
plt.figure(3, figsize=(9,3))
fig, ax = plt.subplots(figsize=(9,3))
plt.plot(xx1, yy1, color = 'black', label = 'obstacle, R = 6m')
plt.plot(true_x, true_y, color = 'red', label = 'Ground truth')
plt.plot(loc_x, loc_y, color = 'blue', label = 'BarrierNet, R = 6m')
# plt.plot(loc_x_6_dfb, loc_y_6_dfb, linestyle='--', color = 'blue', label = 'DFB, R = 6m')
# plt.plot(loc_x_6_fc, loc_y_6_fc, linestyle=':', color = 'blue', label = 'FC, R = 6m')
plt.legend(loc ='lower left', prop={'size': 12}, shadow = False)
plt.ylabel('$y/m$',fontsize=14)
plt.xlabel('$x/m$',fontsize=14)
ax.xaxis.set_label_coords(0.5, -0.06)
plt.axis('equal')
ax.set_rasterized(True)
plt.savefig('./log/bnet{:02d}/test_traj'.format(exp))
# plt.show()


if abnet == 1:
    plt.figure(4)
    fig, ax = plt.subplots()
    plt.plot(tr, p1, color = 'green', label = '$p_1(z)$')
    plt.plot(tr, p2, color = 'blue', label = '$p_2(z)$')
    plt.legend(prop={'size': 16})
    plt.ylabel('Penalty functions',fontsize=14)
    plt.xlabel('time$/s$',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_rasterized(True)
    plt.savefig('./log/bnet{:02d}/test_penalty'.format(exp))
    # plt.show()


xx = [0,12]
yy = [0,0]
plt.figure(5)
fig, ax = plt.subplots()
plt.plot(xx, yy, linestyle='--', color = 'red', label = 'obstacle boundary')
plt.plot(tr, safety, color = 'blue', label = 'BarrierNet, R = 6m')
# plt.plot(tr, safety_6_dfb, linestyle='--', color = 'blue', label = 'DFB, R = 6m')
# plt.plot(tr, safety_6_fc, linestyle=':', color = 'blue', label = 'FC, R = 6m')
plt.ylim([-200, 200])
plt.legend()
plt.ylabel('HOCBF $b(x)$',fontsize=14)
plt.xlabel('time$/s$',fontsize=14)
plt.savefig('./log/bnet{:02d}/test_safety'.format(exp))
# plt.show()

print("end")
