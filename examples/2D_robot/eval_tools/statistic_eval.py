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

write_file = False

#dynamics
def dynamics(y,t):
    dxdt = y[3]*np.cos(y[2])
    dydt = y[3]*np.sin(y[2])
    dttdt = y[4] #u1
    dvdt = y[5]  #u2
    return [dxdt,dydt,dttdt,dvdt, 0, 0]

# CUDA for PyTorch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

loss_fn = nn.MSELoss()

def abnet_test(model_list, wt_list):
    px, py, theta, speed, dsty = init[0], init[1], init[2], init[3], init[4]
    safety, loc_y, loc_xy = [], [], []
    dt = [0,0.1]
    obs_x, obs_y, R = 40, 15, 6

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
            loc_y.append(py)
            loc_xy.append(torch.tensor([[px, py]]))

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
            

            test1.append(torch.tensor([[ctrl[0], ctrl[1]]]))
            
            #update state
            state = [px,py,theta,speed]
            state.append(ctrl[0])
            state.append(ctrl[1])
            
            #update dynamics
            rt = np.float32(odeint(dynamics,state,dt))
            px, py, theta, speed = rt[1][0], rt[1][1], rt[1][2], rt[1][3]

        test1 = torch.cat(test1, dim = 0)
        loc_xy = torch.cat(loc_xy, dim = 0)
        test1_gt = torch.from_numpy(test_labels)
        
        loss = loss_fn(test1, test1_gt)

    return loss.cpu().item(), test1, safety, loc_y, loc_xy


def test_general(model, mode_model, mode_rt):
    px, py, theta, speed, dsty = init[0], init[1], init[2], init[3], init[4]
    safety, loc_y, loc_xy = [], [], []
    dt = [0,0.1]
    obs_x, obs_y, R = 40, 15, 6

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
            loc_y.append(py)
            loc_xy.append(torch.tensor([[px, py]]))

            #prepare for model input
            x_r = Variable(torch.from_numpy(np.array([x,y,tt,v,dst])), requires_grad=False)
            x_r = torch.reshape(x_r, (1,nFeatures))
            x_r = x_r.to(device)
            if mode_model == 1:
                ctrl = model(x_r, 0, 0)
            else:
                ctrl = model(x_r, 0)
            if mode_rt == 1:
                test1.append(torch.tensor([[ctrl[0], ctrl[1]]]))
            else:
                test1.append(torch.tensor([[ctrl[0,0].item(), ctrl[0,1].item()]]))
            #update state
            state = [px,py,theta,speed]
            if mode_rt == 1:
                state.append(ctrl[0])
                state.append(ctrl[1])
            else:
                state.append(ctrl[0,0].item()) 
                state.append(ctrl[0,1].item())
            
            #update dynamics
            rt = np.float32(odeint(dynamics,state,dt))
            px, py, theta, speed = rt[1][0], rt[1][1], rt[1][2], rt[1][3]
            

        test1 = torch.cat(test1, dim = 0)
        loc_xy = torch.cat(loc_xy, dim = 0)
        test1_gt = torch.from_numpy(test_labels)

        temp_y = np.array(loc_y)
        if np.max(temp_y) > 20:
            test1_gt[:,0] = -test1_gt[:,0]
        
        loss = loss_fn(test1, test1_gt)

    return loss.cpu().item(), test1, safety, loc_y, loc_xy


def test_general_sc(model, mode_model, mode_rt):
    px, py, theta, speed, dsty = init[0], init[1], init[2], init[3], init[4]
    safety, loc_y, loc_xy = [], [], []
    dt = [0,0.1]
    obs_x, obs_y, R = 40, 15, 6

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
            loc_y.append(py)
            loc_xy.append(torch.tensor([[px, py]]))

            #prepare for model input
            x_r = Variable(torch.from_numpy(np.array([x,y,tt,v,dst])), requires_grad=False)
            x_r = torch.reshape(x_r, (1,nFeatures))
            x_r = x_r.to(device)

            wt = 1.0/len(model)
            if mode_model == 1:
                ctrl = 0
                for kk in range(len(model)):
                    if kk > 0:
                        model[kk].x32 = model[0].x32
                    ctrl += wt*model[kk](x_r, 0, 0)
            else:
                ctrl = model(x_r, 0)
            if mode_rt == 1:
                test1.append(torch.tensor([[ctrl[0], ctrl[1]]]))
            else:
                test1.append(torch.tensor([[ctrl[0,0].item(), ctrl[0,1].item()]]))
            #update state
            state = [px,py,theta,speed]
            if mode_rt == 1:
                state.append(ctrl[0])
                state.append(ctrl[1])
            else:
                state.append(ctrl[0,0].item()) 
                state.append(ctrl[0,1].item())
            
            #update dynamics
            rt = np.float32(odeint(dynamics,state,dt))
            px, py, theta, speed = rt[1][0], rt[1][1], rt[1][2], rt[1][3]
            

        test1 = torch.cat(test1, dim = 0)
        loc_xy = torch.cat(loc_xy, dim = 0)
        test1_gt = torch.from_numpy(test_labels)

        temp_y = np.array(loc_y)
        if np.max(temp_y) > 20:
            test1_gt[:,0] = -test1_gt[:,0]
        
        loss = loss_fn(test1, test1_gt)

    return loss.cpu().item(), test1, safety, loc_y, loc_xy


from scipy.stats import gaussian_kde
def test_bnet_up(model):
    px, py, theta, speed, dsty = init[0], init[1], init[2], init[3], init[4]
    safety, loc_y, loc_xy = [], [], []
    dt = [0,0.1]
    obs_x, obs_y, R = 40, 15, 6

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
            loc_y.append(py)
            loc_xy.append(torch.tensor([[px, py]]))

            #prepare for model input
            x_r = Variable(torch.from_numpy(np.array([x,y,tt,v,dst])), requires_grad=False)
            x_r = torch.reshape(x_r, (1,nFeatures))
            x_r = x_r.to(device)

            x_samples = []
            for kk in range(10):
                ctrl0 = model(x_r, 0)
                x_samples.append(torch.tensor([[ctrl0[0], ctrl0[1]]]))
            x_samples = torch.cat(x_samples, dim = 0)

            x_samples_np = x_samples.cpu().numpy()
            kde_x1 = gaussian_kde(x_samples_np[:,0])
            kde_x2 = gaussian_kde(x_samples_np[:,1])

            # max probability
            ctrl = torch.stack([x_samples[kde_x1(x_samples_np[:,0]).argmax(),0],
                                x_samples[kde_x2(x_samples_np[:,1]).argmax(),1]])[None,:]
            
            ctrl = ctrl.cpu().numpy()[0]
            

            test1.append(torch.tensor([[ctrl[0], ctrl[1]]]))
            #update state
            state = [px,py,theta,speed]
            state.append(ctrl[0])
            state.append(ctrl[1])
            
            #update dynamics
            rt = np.float32(odeint(dynamics,state,dt))
            px, py, theta, speed = rt[1][0], rt[1][1], rt[1][2], rt[1][3]
            

        test1 = torch.cat(test1, dim = 0)
        loc_xy = torch.cat(loc_xy, dim = 0)
        test1_gt = torch.from_numpy(test_labels)
        
        loss = loss_fn(test1, test1_gt)

    return loss.cpu().item(), test1, safety, loc_y, loc_xy

def test_fc_up(model):
    px, py, theta, speed, dsty = init[0], init[1], init[2], init[3], init[4]
    safety, loc_y, loc_xy = [], [], []
    dt = [0,0.1]
    obs_x, obs_y, R = 40, 15, 6

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
            loc_y.append(py)
            loc_xy.append(torch.tensor([[px, py]]))

            #prepare for model input
            x_r = Variable(torch.from_numpy(np.array([x,y,tt,v,dst])), requires_grad=False)
            x_r = torch.reshape(x_r, (1,nFeatures))
            x_r = x_r.to(device)

            x_samples = []
            for kk in range(10):
                ctrl0 = model(x_r, 0)
                # x_samples.append(ctrl0)
                x_samples.append(ctrl0.unsqueeze(0))
            x_samples = torch.cat(x_samples, dim = 0)

            ctrl = torch.mean(x_samples, dim = 0).cpu().numpy()[0]

            # x_samples_np = x_samples.cpu().numpy()
            # kde_x1 = gaussian_kde(x_samples_np[:,0])
            # kde_x2 = gaussian_kde(x_samples_np[:,1])

            # # max probability
            # ctrl = torch.stack([x_samples[kde_x1(x_samples_np[:,0]).argmax(),0],
            #                     x_samples[kde_x2(x_samples_np[:,1]).argmax(),1]])[None,:]
            
            # ctrl = ctrl.cpu().numpy()[0]
            

            test1.append(torch.tensor([[ctrl[0], ctrl[1]]]))
            #update state
            state = [px,py,theta,speed]
            state.append(ctrl[0])
            state.append(ctrl[1])
            
            #update dynamics
            rt = np.float32(odeint(dynamics,state,dt))
            px, py, theta, speed = rt[1][0], rt[1][1], rt[1][2], rt[1][3]
            

        test1 = torch.cat(test1, dim = 0)
        loc_xy = torch.cat(loc_xy, dim = 0)
        test1_gt = torch.from_numpy(test_labels)
        
        loss = loss_fn(test1, test1_gt)

    return loss.cpu().item(), test1, safety, loc_y, loc_xy


# nFeatures, nHidden1, nHidden21, nHidden22, nCls = 5, 128, 32, 32, 2
# model = models.BarrierNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
# model.load_state_dict(torch.load("log/bnet02/model_bn.pth"))
# model.eval()    

# loss_no_noise, test1_no_noise, safety_no_noise, loc_y_no_noise, loc_xy_no_noise = test_general(model, 0, 1)

# safety_profiles = []
# loss_profiles = []
# vari_profiles = []
# time_profiles = []
# loc_profiles = []
# model.add_noise = True
# import time
# for i in range(100):
#     print(i)
#     start = time.time()
#     loss, test1, safety, loc_y, loc_xy = test_general(model, 0, 1)
#     end = time.time()
#     # loc_y = np.array(loc_y)
#     # if np.max(loc_y) > 20:
#     #     continue
#     time_profiles.append(end-start)
#     safety = np.array(safety)
#     safety_profiles.append(np.min(safety))
#     loss_profiles.append(loss)
#     test1 = test1.unsqueeze(0)
#     vari_profiles.append(test1)
#     loc_profiles.append(loc_xy)

# time_profiles = np.array(time_profiles)
# loss_profiles = np.array(loss_profiles)
# vari_profiles = torch.cat(vari_profiles, dim = 0)
# safety_profiles = np.array(safety_profiles)
# loc_profiles = torch.cat(loc_profiles, dim = 0)

# if write_file:
#     vari_np = vari_profiles.cpu().numpy()
#     np.save('./log/ctrl_bnet.npy', vari_np)

# print("bnet time ave:", np.mean(time_profiles), " bnet time var:", np.var(time_profiles))
# print("bnet loss ave:", np.mean(loss_profiles), " bnet loss var:", np.var(loss_profiles))
# print("bnet Safety:", np.min(safety_profiles))
# print("bnet conser ave:", np.mean(safety_profiles), " bnet conser var:", np.var(safety_profiles))
# print("bnet uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))

# exit()
######################################## scalable training testing
# model1 = models.ABNet_sc(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads = 10).to(device)
# model1.load_state_dict(torch.load("./log/abnet{:02d}".format(3) + "/model_abn.pth"))
# model1.eval()
# model1.first = 1   
# model1.add_noise = True

model1 = models.ABNet_sc(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads = 100).to(device)
model1.load_state_dict(torch.load("./log/abnet{:02d}".format(8) + "/model_abn06.pth"))
model1.eval()
model1.first = 1   
model1.add_noise = True

model2 = models.ABNet_sc(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads = 100).to(device)
model2.load_state_dict(torch.load("./log/abnet{:02d}".format(8) + "/model_abn19.pth"))  #6
model2.eval() 
model2.add_noise = True

model3 = models.ABNet_sc(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads = 100).to(device)
model3.load_state_dict(torch.load("./log/abnet{:02d}".format(7) + "/model_abn19.pth"))
model3.eval() 
model3.add_noise = True

model4 = models.ABNet_sc(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads = 100).to(device)
model4.load_state_dict(torch.load("./log/abnet{:02d}".format(8) + "/model_abn17.pth"))
model4.eval() 
model4.add_noise = True

model5 = models.ABNet_sc(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads = 100).to(device)
model5.load_state_dict(torch.load("./log/abnet{:02d}".format(8) + "/model_abn12.pth"))
model5.eval() 
model5.add_noise = True

model6 = models.ABNet_sc(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads = 100).to(device)
model6.load_state_dict(torch.load("./log/abnet{:02d}".format(8) + "/model_abn18.pth"))
model6.eval() 
model6.add_noise = True

model7 = models.ABNet_sc(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads = 100).to(device)
model7.load_state_dict(torch.load("./log/abnet{:02d}".format(8) + "/model_abn16.pth"))
model7.eval() 
model7.add_noise = True

model8 = models.ABNet_sc(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads = 100).to(device)
model8.load_state_dict(torch.load("./log/abnet{:02d}".format(8) + "/model_abn15.pth"))
model8.eval() 
model8.add_noise = True

model9 = models.ABNet_sc(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads = 100).to(device)
model9.load_state_dict(torch.load("./log/abnet{:02d}".format(6) + "/model_abn18.pth"))
model9.eval() 
model9.add_noise = True

model10 = models.ABNet_sc(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads = 100).to(device)
model10.load_state_dict(torch.load("./log/abnet{:02d}".format(6) + "/model_abn17.pth"))
model10.eval() 
model10.add_noise = True

# model = [model1, model2]
model = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]
# model.append(model1)
# model.append(model2)

# model2 = models.ABNet_sc(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads = 10).to(device)
# model2.load_state_dict(torch.load("./log/abnet{:02d}".format(4) + "/model_abn.pth"))
# model2.eval()    
# model2.add_noise = True

# import pdb; pdb.set_trace()

safety_profiles = []
loss_profiles = []
vari_profiles = []
time_profiles = []
loc_profiles = []
import time
for i in range(100):
    print(i)
    start = time.time()
    loss, test1, safety, loc_y, loc_xy = test_general_sc(model, 1, 0)
    end = time.time()
    # loc_y = np.array(loc_y)
    # if np.max(loc_y) > 20:
    #     continue
    time_profiles.append(end-start)
    safety = np.array(safety)
    safety_profiles.append(np.min(safety))
    loss_profiles.append(loss)
    test1 = test1.unsqueeze(0)
    vari_profiles.append(test1)
    loc_profiles.append(loc_xy)

time_profiles = np.array(time_profiles)
loss_profiles = np.array(loss_profiles)
vari_profiles = torch.cat(vari_profiles, dim = 0)
safety_profiles = np.array(safety_profiles)
loc_profiles = torch.cat(loc_profiles, dim = 0)

# if write_file:
#     vari_np = vari_profiles.cpu().numpy()
#     np.save('./log/ctrl_abnet-10.npy', vari_np)

print("abnet-sc time ave:", np.mean(time_profiles), " time var:", np.var(time_profiles))
print("abnet-sc loss ave:", np.mean(loss_profiles), " loss var:", np.var(loss_profiles))
print("abnet-sc Safety:", np.min(safety_profiles))
print("abnet-sc conser ave:", np.mean(safety_profiles), " conser var:", np.var(safety_profiles))
print("abnet-sc uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))

##################################################
exit()
import pdb; pdb.set_trace()

#####################################3## FC kde and mean
model = models.FCNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
model.load_state_dict(torch.load("log/model_fc.pth"))
model.eval()    

safety_profiles = []
loss_profiles = []
vari_profiles = []
time_profiles = []
loc_profiles = []
import time
model.add_noise = True
for i in range(100):
    print(i)
    start = time.time()
    loss, test1, safety, loc_y, loc_xy = test_fc_up(model)
    end = time.time()
    loc_y = np.array(loc_y)
    # if np.max(loc_y) > 20:            # NOTE here
    #     continue
    time_profiles.append(end-start)
    safety = np.array(safety)
    safety_profiles.append(np.min(safety))
    loss_profiles.append(loss)
    test1 = test1.unsqueeze(0)
    vari_profiles.append(test1)
    loc_profiles.append(loc_xy)

# import pdb; pdb.set_trace()

time_profiles = np.array(time_profiles)
loss_profiles = np.array(loss_profiles)
vari_profiles = torch.cat(vari_profiles, dim = 0)
safety_profiles = np.array(safety_profiles)
loc_profiles = torch.cat(loc_profiles, dim = 0)

vari_np = vari_profiles.cpu().numpy()
np.save('./log/ctrl_fc-mean.npy', vari_np)
    

print("FC time ave:", np.mean(time_profiles), " FC time var:", np.var(time_profiles))
print("FC loss ave:", np.mean(loss_profiles), " FC loss var:", np.var(loss_profiles))
print("FC Safety:", np.min(safety_profiles))
print("FC conser ave:", np.mean(safety_profiles), " FC conser var:", np.var(safety_profiles))
# import pdb; pdb.set_trace()
print("FC uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))

######################################################
exit()
import pdb; pdb.set_trace()

######################################## gaussian-kde
nFeatures, nHidden1, nHidden21, nHidden22, nCls = 5, 128, 32, 32, 2
model = models.BarrierNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
model.load_state_dict(torch.load("log/bnet01/model_bn.pth"))
model.eval()    

safety_profiles = []
loss_profiles = []
vari_profiles = []
time_profiles = []
loc_profiles = []
import time
model.add_noise = True
for i in range(100):
    print(i)
    start = time.time()
    loss, test1, safety, loc_y, loc_xy = test_bnet_up(model)
    end = time.time()
    loc_y = np.array(loc_y)
    if np.max(loc_y) > 20:
        continue
    time_profiles.append(end-start)
    safety = np.array(safety)
    safety_profiles.append(np.min(safety))
    loss_profiles.append(loss)
    test1 = test1.unsqueeze(0)
    vari_profiles.append(test1)
    loc_profiles.append(loc_xy)

time_profiles = np.array(time_profiles)
loss_profiles = np.array(loss_profiles)
vari_profiles = torch.cat(vari_profiles, dim = 0)
safety_profiles = np.array(safety_profiles)
loc_profiles = torch.cat(loc_profiles, dim = 0)


vari_np = vari_profiles.cpu().numpy()
np.save('./log/ctrl_bnet_up.npy', vari_np)

print("bnet time ave:", np.mean(time_profiles), " bnet time var:", np.var(time_profiles))
print("bnet loss ave:", np.mean(loss_profiles), " bnet loss var:", np.var(loss_profiles))
print("bnet Safety:", np.min(safety_profiles))
print("bnet conser ave:", np.mean(safety_profiles), " bnet conser var:", np.var(safety_profiles))
print("bnet uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))


#######################################$



model = models.FCNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
model.load_state_dict(torch.load("log/model_fc.pth"))
model.eval()    

loss_no_noise, test1_no_noise, safety_no_noise, loc_y_no_noise, loc_xy_no_noise = test_general(model, 0, 0)

safety_profiles = []
loss_profiles = []
vari_profiles = []
time_profiles = []
loc_profiles = []
import time
model.add_noise = True
for i in range(100):
    start = time.time()
    loss, test1, safety, loc_y, loc_xy = test_general(model, 0, 0)
    end = time.time()
    loc_y = np.array(loc_y)
    if np.max(loc_y) > 20:
        continue
    time_profiles.append(end-start)
    safety = np.array(safety)
    safety_profiles.append(np.min(safety))
    loss_profiles.append(loss)
    test1 = test1.unsqueeze(0)
    vari_profiles.append(test1)
    loc_profiles.append(loc_xy)

time_profiles = np.array(time_profiles)
loss_profiles = np.array(loss_profiles)
vari_profiles = torch.cat(vari_profiles, dim = 0)
safety_profiles = np.array(safety_profiles)
loc_profiles = torch.cat(loc_profiles, dim = 0)

if write_file:
    vari_np = vari_profiles.cpu().numpy()
    gt = test_labels[:,0:2]
    np.save('./log/ctrl_gt.npy', gt)
    np.save('./log/ctrl_fc.npy', vari_np)
    

print("FC time ave:", np.mean(time_profiles), " FC time var:", np.var(time_profiles))
print("FC loss ave:", np.mean(loss_profiles), " FC loss var:", np.var(loss_profiles))
print("FC Safety:", np.min(safety_profiles))
print("FC conser ave:", np.mean(safety_profiles), " FC conser var:", np.var(safety_profiles))
print("FC uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))
# import pdb; pdb.set_trace()


nFeatures, nHidden1, nHidden21, nHidden22, nCls = 5, 100, 42, 42, 2
model = models.DFBNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
model.load_state_dict(torch.load("log/modelM_DFB.pth"))
model.eval()    

loss_no_noise, test1_no_noise, safety_no_noise, loc_y_no_noise, loc_xy_no_noise = test_general(model, 0, 1)

safety_profiles = []
loss_profiles = []
vari_profiles = []
time_profiles = []
loc_profiles = []
model.add_noise = True
for i in range(100):
    start = time.time()
    loss, test1, safety, loc_y, loc_xy = test_general(model, 0, 1)
    end = time.time()
    loc_y = np.array(loc_y)
    if np.max(loc_y) > 20:
        continue
    time_profiles.append(end-start)
    safety = np.array(safety)
    safety_profiles.append(np.min(safety))
    loss_profiles.append(loss)
    test1 = test1.unsqueeze(0)
    vari_profiles.append(test1)
    loc_profiles.append(loc_xy)

time_profiles = np.array(time_profiles)
loss_profiles = np.array(loss_profiles)
vari_profiles = torch.cat(vari_profiles, dim = 0)
safety_profiles = np.array(safety_profiles)
loc_profiles = torch.cat(loc_profiles, dim = 0)

if write_file:
    vari_np = vari_profiles.cpu().numpy()
    np.save('./log/ctrl_dfb.npy', vari_np)

print("DFB time ave:", np.mean(time_profiles), " DFB time var:", np.var(time_profiles))
print("DFB loss ave:", np.mean(loss_profiles), " DFB loss var:", np.var(loss_profiles))
print("DFB Safety:", np.min(safety_profiles))
print("DFB conser ave:", np.mean(safety_profiles), " DFB conser var:", np.var(safety_profiles))
print("DFB uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))




nFeatures, nHidden1, nHidden21, nHidden22, nCls = 5, 128, 32, 32, 2
model = models.BarrierNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
model.load_state_dict(torch.load("log/bnet01/model_bn.pth"))
model.eval()    

loss_no_noise, test1_no_noise, safety_no_noise, loc_y_no_noise, loc_xy_no_noise = test_general(model, 0, 1)

safety_profiles = []
loss_profiles = []
vari_profiles = []
time_profiles = []
loc_profiles = []
model.add_noise = True
for i in range(100):
    start = time.time()
    loss, test1, safety, loc_y, loc_xy = test_general(model, 0, 1)
    end = time.time()
    loc_y = np.array(loc_y)
    if np.max(loc_y) > 20:
        continue
    time_profiles.append(end-start)
    safety = np.array(safety)
    safety_profiles.append(np.min(safety))
    loss_profiles.append(loss)
    test1 = test1.unsqueeze(0)
    vari_profiles.append(test1)
    loc_profiles.append(loc_xy)

time_profiles = np.array(time_profiles)
loss_profiles = np.array(loss_profiles)
vari_profiles = torch.cat(vari_profiles, dim = 0)
safety_profiles = np.array(safety_profiles)
loc_profiles = torch.cat(loc_profiles, dim = 0)

if write_file:
    vari_np = vari_profiles.cpu().numpy()
    np.save('./log/ctrl_bnet.npy', vari_np)

print("bnet time ave:", np.mean(time_profiles), " bnet time var:", np.var(time_profiles))
print("bnet loss ave:", np.mean(loss_profiles), " bnet loss var:", np.var(loss_profiles))
print("bnet Safety:", np.min(safety_profiles))
print("bnet conser ave:", np.mean(safety_profiles), " bnet conser var:", np.var(safety_profiles))
print("bnet uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))


model_list = []
for i in range(10):
    model = models.BarrierNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
    model.load_state_dict(torch.load("./log/bnet{:02d}".format(i+1) + "/model_bn.pth"))
    model.eval() 
    model.add_noise = True
    model_list.append(model)

loss_list = []
for i in range(10):
    loss, test1, safety, loc_y, loc_xy = test_general(model_list[i], 0, 1)
    loss_list.append(1/loss)
loss_np = np.array(loss_list)
wt_sum = np.sum(loss_np)
wt_list = loss_np/wt_sum

safety_profiles = []
loss_profiles = []
vari_profiles = []
time_profiles = []
loc_profiles = []
for i in range(100):
    start = time.time()
    loss, test1, safety, loc_y, loc_xy = abnet_test(model_list, wt_list)
    end = time.time()
    loc_y = np.array(loc_y)
    if np.max(loc_y) > 20:
        continue
    time_profiles.append(end-start)
    safety = np.array(safety)
    safety_profiles.append(np.min(safety))
    loss_profiles.append(loss)
    test1 = test1.unsqueeze(0)
    vari_profiles.append(test1)
    loc_profiles.append(loc_xy)

time_profiles = np.array(time_profiles)
loss_profiles = np.array(loss_profiles)
vari_profiles = torch.cat(vari_profiles, dim = 0)
safety_profiles = np.array(safety_profiles)
loc_profiles = torch.cat(loc_profiles, dim = 0)

if write_file:
    vari_np = vari_profiles.cpu().numpy()
    np.save('./log/ctrl_preabnet-10.npy', vari_np)

print("pre-abnet-10 time ave:", np.mean(time_profiles), " time var:", np.var(time_profiles))
print("pre-abnet-10 loss ave:", np.mean(loss_profiles), " loss var:", np.var(loss_profiles))
print("pre-abnet-10 Safety:", np.min(safety_profiles))
print("pre-abnet-10 conser ave:", np.mean(safety_profiles), " conser var:", np.var(safety_profiles))
print("pre-abnet-10 uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))


model = models.ABNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads = 10).to(device)
model.load_state_dict(torch.load("./log/abnet{:02d}".format(4) + "/model_abn.pth"))
model.eval()    

loss_no_noise, test1_no_noise, safety_no_noise, loc_y_no_noise, loc_xy_no_noise = test_general(model, 1, 0)

safety_profiles = []
loss_profiles = []
vari_profiles = []
time_profiles = []
loc_profiles = []
model.add_noise = True
for i in range(100):
    start = time.time()
    loss, test1, safety, loc_y, loc_xy = test_general(model, 1, 0)
    end = time.time()
    loc_y = np.array(loc_y)
    if np.max(loc_y) > 20:
        continue
    time_profiles.append(end-start)
    safety = np.array(safety)
    safety_profiles.append(np.min(safety))
    loss_profiles.append(loss)
    test1 = test1.unsqueeze(0)
    vari_profiles.append(test1)
    loc_profiles.append(loc_xy)

time_profiles = np.array(time_profiles)
loss_profiles = np.array(loss_profiles)
vari_profiles = torch.cat(vari_profiles, dim = 0)
safety_profiles = np.array(safety_profiles)
loc_profiles = torch.cat(loc_profiles, dim = 0)

if write_file:
    vari_np = vari_profiles.cpu().numpy()
    np.save('./log/ctrl_abnet-10.npy', vari_np)

print("abnet-10 time ave:", np.mean(time_profiles), " time var:", np.var(time_profiles))
print("abnet-10 loss ave:", np.mean(loss_profiles), " loss var:", np.var(loss_profiles))
print("abnet-10 Safety:", np.min(safety_profiles))
print("abnet-10 conser ave:", np.mean(safety_profiles), " conser var:", np.var(safety_profiles))
print("abnet-10 uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))



model = models.ABNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads = 100).to(device)
model.load_state_dict(torch.load("./log/abnet{:02d}".format(8) + "/model_abn06.pth"))
model.eval()    

loss_no_noise, test1_no_noise, safety_no_noise, loc_y_no_noise, loc_xy_no_noise = test_general(model, 1, 0)

safety_profiles = []
loss_profiles = []
vari_profiles = []
time_profiles = []
loc_profiles = []
model.add_noise = True
for i in range(100):
    start = time.time()
    loss, test1, safety, loc_y, loc_xy = test_general(model, 1, 0)
    end = time.time()
    loc_y = np.array(loc_y)
    if np.max(loc_y) > 20:
        continue
    time_profiles.append(end-start)
    safety = np.array(safety)
    safety_profiles.append(np.min(safety))
    loss_profiles.append(loss)
    test1 = test1.unsqueeze(0)
    vari_profiles.append(test1)
    loc_profiles.append(loc_xy)

time_profiles = np.array(time_profiles)
loss_profiles = np.array(loss_profiles)
vari_profiles = torch.cat(vari_profiles, dim = 0)
safety_profiles = np.array(safety_profiles)
loc_profiles = torch.cat(loc_profiles, dim = 0)

if write_file:
    vari_np = vari_profiles.cpu().numpy()
    np.save('./log/ctrl_abnet-100.npy', vari_np)

print("abnet-100 time ave:", np.mean(time_profiles), " time var:", np.var(time_profiles))
print("abnet-100 loss ave:", np.mean(loss_profiles), " loss var:", np.var(loss_profiles))
print("abnet-100 Safety:", np.min(safety_profiles))
print("abnet-100 conser ave:", np.mean(safety_profiles), " conser var:", np.var(safety_profiles))
print("abnet-100 uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))


# ctrl1 = np.array(ctrl1)    
# ctrl1.tofile('ctrlM1_r12_dfb.dat')

# ctrl1_6 = np.fromfile('ctrl1_r6.dat', dtype=np.float64)
# ctrl1_6 = ctrl1_6.tolist()