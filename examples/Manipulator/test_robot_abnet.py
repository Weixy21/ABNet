import torch
import torch.nn as nn
import scipy.io as sio
import models
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
plt.style.use('bmh')
from scipy.integrate import odeint
import matplotlib.animation as animation
#export PYTHONNOUSERSITE=True   for python
#export PYOPENGL_PLATFORM=egl   for Vista (cannot connect to "%s"' % name)

#choose a barriernet or not 1-barriernet
barriernet = 0
abnet = 1
exp = 1 

#dynamics
def dynamics(y,t):
    dxdt = y[1]
    dydt = y[4]  #u1
    dttdt = y[3] 
    dvdt = y[5]  #u2
    return [dxdt,dydt,dttdt,dvdt, 0, 0]

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using {} device".format(device))
torch.backends.cudnn.benchmark = True

# Datasets
train_data = sio.loadmat('data/data_train.mat') 
train_data = train_data['data']
valid_data = sio.loadmat('data/data_valid.mat') 
valid_data = valid_data['data']
test_data = sio.loadmat('data/data_test.mat') 
test_data = test_data['data']

train0 = np.double(train_data[:,0:6])  # theta1, w1, theta2, w2, dst_x, dst_y
train_labels_unnorm = np.reshape(np.double(train_data[:,6:8]), (len(train_data),2)) #w1_derivative, w2_derivative
valid0 = np.double(valid_data[:,0:6]) 
valid_labels_unnorm = np.reshape(np.double(valid_data[:,6:8]), (len(valid_data),2))
test0 = np.double(test_data[:,0:6]) 
test_labels_unnorm = np.reshape(np.double(test_data[:,6:8]), (len(test_data),2))
init = test0[0]

mean = np.mean(train0, axis = 0)
std = np.std(train0, axis = 0)

mean_label = np.mean(train_labels_unnorm, axis = 0)
std_label = np.std(train_labels_unnorm, axis = 0)


# train0 = (train0 - mean)/std
# valid0 = (valid0 - mean)/std
# test0 = (test0 - mean)/std

# train_labels = (train_labels_unnorm - mean_label)/std_label
# valid_labels = (valid_labels_unnorm - mean_label)/std_label
# test_labels = (test_labels_unnorm - mean_label)/std_label


def map(obs_x = 0, obs_y = 7):  # draw background and return plot_handles
    fig, ax = plt.subplots(figsize=(12,8))
    ax.axis('equal')
    ax.set_facecolor("white")

    r = 3.8 
    theta = np.linspace(0, 2*3.14, 100)
    x = r*np.cos(theta) 
    y = r*np.sin(theta)
    rho = r*np.sin(theta)
    ax.plot(x + obs_x, y + obs_y,'k-')
    ax.fill(x + obs_x, y + obs_y,'c')

    ax.plot(init[4],init[5], color = 'k', marker = '+', markeredgecolor = 'b', markerfacecolor = 'b', linewidth = 5)
    ax.text(init[4],init[5]+1, 'dst', color = 'b', fontsize = 20)

    p1, = ax.plot([],[], color = 'r', linewidth = 10)
    p2, = ax.plot([],[], color = 'g', linewidth = 10)
    
    p3, = ax.plot([],[],color = 'k',marker = 'v', markeredgecolor = 'k', markerfacecolor = 'k', linewidth = 5)
    p4, = ax.plot([],[],color = 'k',marker = 'v', markeredgecolor = 'k', markerfacecolor = 'k', linewidth = 5)
    
    p5, = ax.plot([],[],color = 'g', linewidth = 20)
    p6, = ax.plot([],[],color = 'w', linewidth = 10)
    
    p7, = ax.plot([],[],color = 'm',marker = 'o', markeredgecolor = 'm', markerfacecolor = 'm', linewidth = 5)
    
    ax.axis([-8, 8, -2, 10])

    plt.show(block = False)
    plt.pause(0.02)
    return ax, fig, [p1, p2, p3, p4, p5, p6, p7]


# Initialize the model.
model_param = [6, 128, 256, 128, 128, 32, 32, 2] 
if barriernet == 1:
    model = models.BarrierNet(model_param, mean, std, mean_label, std_label, device, bn=False, activation = 'relu').to(device)
    model.load_state_dict(torch.load("./log/bnet{:02d}".format(exp) + "/model_bn09.pth"))
elif abnet == 1:
    model = models.ABNet(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 10).to(device)
    model.load_state_dict(torch.load("./log/abnet{:02d}".format(exp) + "/model_abn09.pth"))
    model.use_cf = True  # closed form   
else:
    model = models.DFBNet(model_param, mean, std, mean_label, std_label, device, bn=False).to(device)
    model.load_state_dict(torch.load("./log/fcnet{:02d}/model_fc02.pth".format(exp)))
model.eval()    
model.add_noise = True


l1, l2 = 3, 3
theta1, w1, theta2, w2, dstx, dsty = init[0], init[1], init[2], init[3], init[4], init[5]
tr, tr0 = [], 0
ctrl1, ctrl2, safety, ctrl1_real, ctrl2_real, p1, p2, loc_x, loc_y = [], [], [], [], [], [], [], [], []
dt = [0,0.1]
obs_x, obs_y, R = 0, 7, 4
true_x, true_y = [], []

loss_fn = nn.MSELoss()
test1, test2 = [], []

# running on a vehicle
with torch.no_grad():
    for i in range(0,len(test0),1): #train0, 10
        #normalize
        t1 = (theta1 - mean[0])/std[0]
        o1 = (w1 - mean[1])/std[1]
        t2 = (theta2 - mean[2])/std[2]
        o2 = (w2 - mean[3])/std[3]
        dx = (dstx - mean[4])/std[4]
        dy = (dsty - mean[5])/std[5]
        
        #get safety metric
        safe = (l1*np.cos(theta1) + l2*np.cos(theta2) - obs_x)**2 + (l1*np.sin(theta1) + l2*np.sin(theta2) - obs_y)**2 - R**2
        safety.append(safe)
        loc_x.append(l1*np.cos(theta1) + l2*np.cos(theta2))
        loc_y.append(l1*np.sin(theta1) + l2*np.sin(theta2))
        true_x.append(l1*np.cos(test0[i][0])+l2*np.cos(test0[i][2]))
        true_y.append(l1*np.sin(test0[i][0])+l2*np.sin(test0[i][2]))

        #prepare for model input
        x_r = Variable(torch.from_numpy(np.array([t1,o1,t2,o2,dx,dy])), requires_grad=False)
        x_r = torch.reshape(x_r, (1,model_param[0]))
        x_r = x_r.to(device)
        ctrl = model(x_r, 0)  #already unnormalized
        
        
        #get the penalty functions
        if barriernet == 1:
            p1.append(model.p1.item())  #only for the barriernet
            p2.append(model.p2.item())
        
        #update state
        state = [theta1,w1,theta2,w2]
        state.append(ctrl[0,0].item()) 
        state.append(ctrl[0,1].item())

        test1.append(torch.tensor([[ctrl[0,0].item(), ctrl[0,1].item()]]))
        test2.append(torch.tensor([[ctrl[0,1].item(), ctrl[0,0].item()]]))
        
        #update dynamics
        rt = np.float32(odeint(dynamics,state,dt))
        theta1, w1, theta2, w2 = rt[1][0], rt[1][1], rt[1][2], rt[1][3]
        

        ctrl1.append(ctrl[0,0].item()) 
        ctrl2.append(ctrl[0,1].item())
        ctrl1_real.append(test_labels_unnorm[i][0])
        ctrl2_real.append(test_labels_unnorm[i][1])
        tr.append(tr0)
        tr0 = tr0 + dt[1]
    test1 = torch.cat(test1, dim = 0)
    test2 = torch.cat(test2, dim = 0)
    test1_gt = torch.from_numpy(test_labels_unnorm)
    
    loss1 = loss_fn(test1, test1_gt)
    loss2 = loss_fn(test2, test1_gt)
    print('testing loss 1:', loss1.cpu().item(), 'testing loss 2:', loss2.cpu().item())

    if loss1.cpu().item() <= loss2.cpu().item():
        test = test1.cpu().numpy()
    else:
        test = test2.cpu().numpy()
print("Implementation done!")

plt.figure(1)
fig, ax = plt.subplots()
plt.plot(tr, ctrl1_real, color = 'red', label = 'Ground truth')
plt.plot(tr, test[:,0], color = 'blue', label = 'BarrierNet, R = 6m')  #ctrl1
# plt.plot(tr, ctrl1_6_dfb, linestyle='--', color = 'blue', label = 'DFB, R = 6m')
# plt.plot(tr, ctrl1_6_fc, linestyle=':', color = 'blue', label = 'FC, R = 6m')
plt.legend(loc ='upper left', prop={'size': 10})
plt.ylabel('Joint 1 $u_1/(rad/s^2)$',fontsize=14)
plt.xlabel('time$/s$',fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.set_rasterized(True)
if barriernet == 1:
    plt.savefig('./log/bnet{:02d}/test_u1'.format(exp))
elif abnet == 1:
    plt.savefig('./log/abnet{:02d}/test_u1'.format(exp))
else:
    plt.savefig('./log/fcnet{:02d}/test_u1'.format(exp))   
# plt.show()

plt.figure(2)
fig, ax = plt.subplots()
plt.plot(tr, ctrl2_real, color = 'red', label = 'Ground truth')
plt.plot(tr, test[:,1], color = 'blue', label = 'BarrierNet, R = 6m')  # ctrl2
# plt.plot(tr, ctrl2_6_dfb, linestyle='--', color = 'blue', label = 'DFB, R = 6m')
# plt.plot(tr, ctrl2_6_fc, linestyle=':', color = 'blue', label = 'FC, R = 6m')
plt.legend(prop={'size': 12})
plt.ylabel('Joint 2 $u_2/(rad/s^2)$',fontsize=14)
plt.xlabel('time$/s$',fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.set_rasterized(True)
if barriernet == 1:
    plt.savefig('./log/bnet{:02d}/test_u2'.format(exp))
elif abnet == 1:
    plt.savefig('./log/abnet{:02d}/test_u2'.format(exp))
else:
    plt.savefig('./log/fcnet{:02d}/test_u2'.format(exp))    
# plt.show()


tt = np.linspace(0, 2*3.14, 60)
r = 4
xx1 = r*np.cos(tt) + 0
yy1 = r*np.sin(tt) + 7
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
if barriernet == 1:
    plt.savefig('./log/bnet{:02d}/test_traj'.format(exp))
elif abnet == 1:
    plt.savefig('./log/abnet{:02d}/test_traj'.format(exp))
else:
    plt.savefig('./log/fcnet{:02d}/test_traj'.format(exp))  
# plt.show()


if barriernet == 1:
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
    if abnet == 1:
        plt.savefig('./log/abnet{:02d}/test_penalty'.format(exp))
    else:
        plt.savefig('./log/bnet{:02d}/test_penalty'.format(exp))
    # plt.show()


xx = [0,35]
yy = [0,0]
plt.figure(5)
fig, ax = plt.subplots()
plt.plot(xx, yy, linestyle='--', color = 'red', label = 'obstacle boundary')
plt.plot(tr, safety, color = 'blue', label = 'BarrierNet, R = 6m')
# plt.plot(tr, safety_6_dfb, linestyle='--', color = 'blue', label = 'DFB, R = 6m')
# plt.plot(tr, safety_6_fc, linestyle=':', color = 'blue', label = 'FC, R = 6m')
plt.ylim([-20, 100])
plt.legend()
plt.ylabel('HOCBF $b(x)$',fontsize=14)
plt.xlabel('time$/s$',fontsize=14)
if barriernet == 1:
    plt.savefig('./log/bnet{:02d}/test_safety'.format(exp))
elif abnet == 1:
    plt.savefig('./log/abnet{:02d}/test_safety'.format(exp))
else:
    plt.savefig('./log/fcnet{:02d}/test_safety'.format(exp))
# plt.show()

print(np.min(np.array(safety), axis = 0))

ax, fig, pen = map()

theta1, w1, theta2, w2, dstx, dsty = init[0], init[1], init[2], init[3], init[4], init[5]

def draw(x1):
    l1, l2 = 3, 3
    px = l1*np.cos(x1[0]) 
    py = l1*np.sin(x1[0])
    pen[0].set_data([0, px], [0, py])
    
    px2 = l1*np.cos(x1[0]) + l2*np.cos(x1[2])
    py2 = l1*np.sin(x1[0]) + l2*np.sin(x1[2])
    pen[1].set_data([px, px2], [py, py2])
    
    
    px21 = l1*np.cos(x1[0]) + 0.9*l2*np.cos(x1[2])
    py21 = l1*np.sin(x1[0]) + 0.9*l2*np.sin(x1[2])
    px22 = l1*np.cos(x1[0]) + 0.93*l2*np.cos(x1[2])
    py22 = l1*np.sin(x1[0]) + 0.93*l2*np.sin(x1[2])
    pen[4].set_data([px21, px22], [py21, py22])
    px23 = l1*np.cos(x1[0]) + 1*l2*np.cos(x1[2])
    py23 = l1*np.sin(x1[0]) + 1*l2*np.sin(x1[2])
    pen[5].set_data([px21, px23], [py21, py23])
    
    pen[2].set_data(0, 0)
    pen[3].set_data(px, py)
    
    pen[6].set_data(px2, py2)

    ax.axis([-8, 8, -2, 10])

# running on a vehicle
def update(n):
    global theta1, w1, theta2, w2
    with torch.no_grad():
        #normalize
        t1 = (theta1 - mean[0])/std[0]
        o1 = (w1 - mean[1])/std[1]
        t2 = (theta2 - mean[2])/std[2]
        o2 = (w2 - mean[3])/std[3]
        dx = (dstx - mean[4])/std[4]
        dy = (dsty - mean[5])/std[5]
        
        draw([theta1,w1,theta2,w2])

        #prepare for model input
        x_r = Variable(torch.from_numpy(np.array([t1,o1,t2,o2,dx,dy])), requires_grad=False)
        x_r = torch.reshape(x_r, (1,model_param[0]))
        x_r = x_r.to(device)
        ctrl = model(x_r, 0)  #already unnormalized
        #update state
        state = [theta1,w1,theta2,w2]
        state.append(ctrl[0,0].item()) 
        state.append(ctrl[0,1].item())

        #update dynamics
        rt = np.float32(odeint(dynamics,state,dt))
        theta1, w1, theta2, w2 = rt[1][0], rt[1][1], rt[1][2], rt[1][3]

ani = animation.FuncAnimation(fig, update, len(test0), fargs=[],
                              interval=25, blit=False, repeat = False)  # interval/ms, blit = False/without return

if barriernet == 1:
    ani.save('./log/bnet{:02d}/test_video.mp4'.format(exp))
elif abnet == 1:
    ani.save('./log/abnet{:02d}/test_video.mp4'.format(exp))
else:
    ani.save('./log/fcnet{:02d}/test_video.mp4'.format(exp))




print("end")
