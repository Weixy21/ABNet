import torch
import torch.nn as nn
import scipy.io as sio
import models
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
plt.style.use('bmh')
from scipy.integrate import odeint
#export PYTHONNOUSERSITE=True   for python
#export PYOPENGL_PLATFORM=egl   for Vista (cannot connect to "%s"' % name)

write_file = False

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

device = "cpu"

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


# Initialize the model param.
model_param = [6, 128, 256, 128, 128, 32, 32, 2] 

loss_fn = nn.MSELoss()

def abnet_test(model_list, wt_list):
    l1, l2 = 3, 3
    theta1, w1, theta2, w2, dstx, dsty = init[0], init[1], init[2], init[3], init[4], init[5]
    safety, loc_xy = [], []
    dt = [0,0.1]
    obs_x, obs_y, R = 0, 7, 4

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
            loc_xy.append(torch.tensor([[l1*np.cos(theta1) + l2*np.cos(theta2), l1*np.sin(theta1) + l2*np.sin(theta2)]]))

            #prepare for model input
            x_r = Variable(torch.from_numpy(np.array([t1,o1,t2,o2,dx,dy])), requires_grad=False)
            x_r = torch.reshape(x_r, (1,model_param[0]))
            x_r = x_r.to(device)
            ctrl = 0
            for k in range(10):
                if k == 0:
                    model_list[k].abnet = True
                else:
                    model_list[k].abnet = False
                    model_list[k].x52 = model_list[0].x52
                ctrl = ctrl + model_list[k](x_r, 0)*wt_list[k]
            

            test1.append(torch.tensor([[ctrl[0,0].item(), ctrl[0,1].item()]]))
            test2.append(torch.tensor([[ctrl[0,1].item(), ctrl[0,0].item()]]))
            
            #update state
            state = [theta1,w1,theta2,w2]
            state.append(ctrl[0,0].item()) 
            state.append(ctrl[0,1].item())
            
            #update dynamics
            rt = np.float32(odeint(dynamics,state,dt))
            theta1, w1, theta2, w2 = rt[1][0], rt[1][1], rt[1][2], rt[1][3]

        test1 = torch.cat(test1, dim = 0)
        test2 = torch.cat(test2, dim = 0)
        test1_gt = torch.from_numpy(test_labels_unnorm)    

        loc_xy = torch.cat(loc_xy, dim = 0)
        
        loss1 = loss_fn(test1, test1_gt)
        loss2 = loss_fn(test2, test1_gt)

        if loss1.cpu().item() <= loss2.cpu().item():
            test = test1
            loss = loss1.cpu().item()
        else:
            test = test2
            loss = loss2.cpu().item()

    return loss, test, safety, loc_xy


def test_general(model):
    l1, l2 = 3, 3
    theta1, w1, theta2, w2, dstx, dsty = init[0], init[1], init[2], init[3], init[4], init[5]
    safety, loc_xy = [], []
    dt = [0,0.1]
    obs_x, obs_y, R = 0, 7, 4

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
            loc_xy.append(torch.tensor([[l1*np.cos(theta1) + l2*np.cos(theta2), l1*np.sin(theta1) + l2*np.sin(theta2)]]))

            #prepare for model input
            x_r = Variable(torch.from_numpy(np.array([t1,o1,t2,o2,dx,dy])), requires_grad=False)
            x_r = torch.reshape(x_r, (1,model_param[0]))
            x_r = x_r.to(device)
            ctrl = model(x_r, 0)

            test1.append(torch.tensor([[ctrl[0,0].item(), ctrl[0,1].item()]]))
            test2.append(torch.tensor([[ctrl[0,1].item(), ctrl[0,0].item()]]))

            #update state
            state = [theta1,w1,theta2,w2]
            state.append(ctrl[0,0].item()) 
            state.append(ctrl[0,1].item())
      
            #update dynamics
            rt = np.float32(odeint(dynamics,state,dt))
            theta1, w1, theta2, w2 = rt[1][0], rt[1][1], rt[1][2], rt[1][3]
            

        test1 = torch.cat(test1, dim = 0)
        test2 = torch.cat(test2, dim = 0)
        test1_gt = torch.from_numpy(test_labels_unnorm)
        loc_xy = torch.cat(loc_xy, dim = 0)
        
        loss1 = loss_fn(test1, test1_gt)
        loss2 = loss_fn(test2, test1_gt)

        if loss1.cpu().item() <= loss2.cpu().item():
            test = test1
            loss = loss1.cpu().item()
        else:
            test = test2
            loss = loss2.cpu().item()

    return loss, test, safety, loc_xy

from scipy.stats import gaussian_kde
def test_bnet_up(model):
    l1, l2 = 3, 3
    theta1, w1, theta2, w2, dstx, dsty = init[0], init[1], init[2], init[3], init[4], init[5]
    safety, loc_xy = [], []
    dt = [0,0.1]
    obs_x, obs_y, R = 0, 7, 4

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
            loc_xy.append(torch.tensor([[l1*np.cos(theta1) + l2*np.cos(theta2), l1*np.sin(theta1) + l2*np.sin(theta2)]]))

            #prepare for model input
            x_r = Variable(torch.from_numpy(np.array([t1,o1,t2,o2,dx,dy])), requires_grad=False)
            x_r = torch.reshape(x_r, (1,model_param[0]))
            x_r = x_r.to(device)

            x_samples = []
            for kk in range(10):
                ctrl0 = model(x_r, 0)
                # x_samples.append(ctrl0)
                x_samples.append(ctrl0.unsqueeze(0))
            x_samples = torch.cat(x_samples, dim = 0)

            ctrl = torch.mean(x_samples, dim = 0)

            # x_samples_np = x_samples.cpu().numpy()
            # kde_x1 = gaussian_kde(x_samples_np[:,0])
            # kde_x2 = gaussian_kde(x_samples_np[:,1])

            # # max probability
            # ctrl = torch.stack([x_samples[kde_x1(x_samples_np[:,0]).argmax(),0],
            #                     x_samples[kde_x2(x_samples_np[:,1]).argmax(),1]])[None,:]

            test1.append(torch.tensor([[ctrl[0,0].item(), ctrl[0,1].item()]]))
            test2.append(torch.tensor([[ctrl[0,1].item(), ctrl[0,0].item()]]))

            #update state
            state = [theta1,w1,theta2,w2]
            state.append(ctrl[0,0].item()) 
            state.append(ctrl[0,1].item())
      
            #update dynamics
            rt = np.float32(odeint(dynamics,state,dt))
            theta1, w1, theta2, w2 = rt[1][0], rt[1][1], rt[1][2], rt[1][3]
            

        test1 = torch.cat(test1, dim = 0)
        test2 = torch.cat(test2, dim = 0)
        test1_gt = torch.from_numpy(test_labels_unnorm)
        loc_xy = torch.cat(loc_xy, dim = 0)
        
        loss1 = loss_fn(test1, test1_gt)
        loss2 = loss_fn(test2, test1_gt)

        if loss1.cpu().item() <= loss2.cpu().item():
            test = test1
            loss = loss1.cpu().item()
        else:
            test = test2
            loss = loss2.cpu().item()

    return loss, test, safety, loc_xy



def test_general_sc(model):
    l1, l2 = 3, 3
    theta1, w1, theta2, w2, dstx, dsty = init[0], init[1], init[2], init[3], init[4], init[5]
    safety, loc_xy = [], []
    dt = [0,0.1]
    obs_x, obs_y, R = 0, 7, 4

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
            loc_xy.append(torch.tensor([[l1*np.cos(theta1) + l2*np.cos(theta2), l1*np.sin(theta1) + l2*np.sin(theta2)]]))

            #prepare for model input
            x_r = Variable(torch.from_numpy(np.array([t1,o1,t2,o2,dx,dy])), requires_grad=False)
            x_r = torch.reshape(x_r, (1,model_param[0]))
            x_r = x_r.to(device)


            wt = 1.0/len(model)
            ctrl = 0
            for kk in range(len(model)):
                if kk > 0:
                    model[kk].x52 = model[0].x52
                ctrl += wt*model[kk](x_r, 0)
            # ctrl = model(x_r, 0)

            test1.append(torch.tensor([[ctrl[0,0].item(), ctrl[0,1].item()]]))
            test2.append(torch.tensor([[ctrl[0,1].item(), ctrl[0,0].item()]]))

            #update state
            state = [theta1,w1,theta2,w2]
            state.append(ctrl[0,0].item()) 
            state.append(ctrl[0,1].item())
      
            #update dynamics
            rt = np.float32(odeint(dynamics,state,dt))
            theta1, w1, theta2, w2 = rt[1][0], rt[1][1], rt[1][2], rt[1][3]
            

        test1 = torch.cat(test1, dim = 0)
        test2 = torch.cat(test2, dim = 0)
        test1_gt = torch.from_numpy(test_labels_unnorm)
        loc_xy = torch.cat(loc_xy, dim = 0)
        
        loss1 = loss_fn(test1, test1_gt)
        loss2 = loss_fn(test2, test1_gt)

        if loss1.cpu().item() <= loss2.cpu().item():
            test = test1
            loss = loss1.cpu().item()
        else:
            test = test2
            loss = loss2.cpu().item()

    return loss, test, safety, loc_xy



# model = models.BarrierNet(model_param, mean, std, mean_label, std_label, device, bn=False, activation = 'relu').to(device)
# model.load_state_dict(torch.load("./log/bnet{:02d}".format(2) + "/model_bn09.pth"))
# model.eval()    

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
#     loss, test1, safety, loc_xy = test_general(model)
#     end = time.time()
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

# print("bnet time ave:", np.mean(time_profiles), " bnet time var:", np.var(time_profiles))
# print("bnet loss ave:", np.mean(loss_profiles), " bnet loss var:", np.var(loss_profiles))
# print("bnet Safety:", np.min(safety_profiles))
# print("bnet conser ave:", np.mean(safety_profiles), " bnet conser var:", np.var(safety_profiles))
# print("bnet uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))

import datetime
ct0 = datetime.datetime.now()

######################################scale training testing
model1 = models.ABNet_sc(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 100).to(device)
model1.load_state_dict(torch.load("./log/abnet{:02d}".format(3) + "/model_abn04.pth"))
model1.eval()
model1.add_noise = True
model1.first = 1

model2 = models.ABNet_sc(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 100).to(device)
model2.load_state_dict(torch.load("./log/abnet{:02d}".format(4) + "/model_abn05.pth"))
model2.eval()
model2.add_noise = True

model3 = models.ABNet_sc(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 100).to(device)
model3.load_state_dict(torch.load("./log/abnet{:02d}".format(4) + "/model_abn04.pth"))
model3.eval()
model3.add_noise = True

model4 = models.ABNet_sc(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 100).to(device)
model4.load_state_dict(torch.load("./log/abnet{:02d}".format(3) + "/model_abn09.pth"))
model4.eval()
model4.add_noise = True

model5 = models.ABNet_sc(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 100).to(device)
model5.load_state_dict(torch.load("./log/abnet{:02d}".format(3) + "/model_abn07.pth"))
model5.eval()
model5.add_noise = True

model6 = models.ABNet_sc(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 100).to(device)
model6.load_state_dict(torch.load("./log/abnet{:02d}".format(4) + "/model_abn03.pth")) # 3, 8
model6.eval()
model6.add_noise = True

model7 = models.ABNet_sc(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 100).to(device)
model7.load_state_dict(torch.load("./log/abnet{:02d}".format(3) + "/model_abn06.pth"))
model7.eval()
model7.add_noise = True

# model8 = models.ABNet_sc(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 100).to(device)
# model8.load_state_dict(torch.load("./log/abnet{:02d}".format(5) + "/model_abn09.pth"))
# model8.eval()
# model8.add_noise = True

# model9 = models.ABNet_sc(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 100).to(device)
# model9.load_state_dict(torch.load("./log/abnet{:02d}".format(5) + "/model_abn08.pth"))  
# model9.eval()
# model9.add_noise = True

# model10 = models.ABNet_sc(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 100).to(device)
# model10.load_state_dict(torch.load("./log/abnet{:02d}".format(5) + "/model_abn07.pth"))  
# model10.eval()
# model10.add_noise = True

model8 = models.ABNet_sc(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 100).to(device)
model8.load_state_dict(torch.load("./log/abnet{:02d}".format(6) + "/model_abn09.pth"))
model8.eval()
model8.add_noise = True

model9 = models.ABNet_sc(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 100).to(device)
model9.load_state_dict(torch.load("./log/abnet{:02d}".format(6) + "/model_abn05.pth"))   # 8,7,6      # 5
model9.eval()
model9.add_noise = True

model10 = models.ABNet_sc(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 100).to(device)
model10.load_state_dict(torch.load("./log/abnet{:02d}".format(6) + "/model_abn03.pth"))  # 7, 6       # 4
model10.eval()
model10.add_noise = True

# model1 = models.ABNet_sc(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 10).to(device)
# model1.load_state_dict(torch.load("./log/abnet{:02d}".format(1) + "/model_abn09.pth"))
# model1.eval()
# model1.first = 1
# model1.add_noise = True

model = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]

# model = [model1]


safety_profiles = []
loss_profiles = []
vari_profiles = []
time_profiles = []
loc_profiles = []
import time
for i in range(100):
    print(i)
    start = time.time()
    loss, test1, safety, loc_xy = test_general_sc(model)
    end = time.time()
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

print("abnet-sc time ave:", np.mean(time_profiles), " time var:", np.var(time_profiles))
print("abnet-sc loss ave:", np.mean(loss_profiles), " loss var:", np.var(loss_profiles))
print("abnet-sc Safety:", np.min(safety_profiles))
print("abnet-sc conser ave:", np.mean(safety_profiles), " conser var:", np.var(safety_profiles))
print("abnet-sc uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))



ct = datetime.datetime.now()
print("Start time:", ct0, "Finish time:", ct)

##############################################
exit()
 
###################################fc up and mean
model = models.FCNet(model_param, mean, std, mean_label, std_label, device, bn=False).to(device)
model.load_state_dict(torch.load("./log/old/fcnet{:02d}/model_fc04.pth".format(1)))
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
    loss, test1, safety, loc_xy = test_bnet_up(model)
    end = time.time()
    
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
np.save('./log/ctrl_fc-mean.npy', vari_np)
np.save('./log/loc_fc-mean.npy', loc_profiles.cpu().numpy())
    

print("FC time ave:", np.mean(time_profiles), " FC time var:", np.var(time_profiles))
print("FC loss ave:", np.mean(loss_profiles), " FC loss var:", np.var(loss_profiles))
print("FC Safety:", np.min(safety_profiles))
print("FC conser ave:", np.mean(safety_profiles), " FC conser var:", np.var(safety_profiles))
print("FC uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))

###################################
exit()
import pdb; pdb.set_trace()
####################################bnet up
model = models.BarrierNet(model_param, mean, std, mean_label, std_label, device, bn=False, activation = 'relu').to(device)
model.load_state_dict(torch.load("./log/bnet{:02d}".format(1) + "/model_bn09.pth"))
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
    loss, test1, safety, loc_xy = test_bnet_up(model)
    end = time.time()
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
np.save('./log/loc_bnet_up.npy', loc_profiles.cpu().numpy())

print("bnet time ave:", np.mean(time_profiles), " bnet time var:", np.var(time_profiles))
print("bnet loss ave:", np.mean(loss_profiles), " bnet loss var:", np.var(loss_profiles))
print("bnet Safety:", np.min(safety_profiles))
print("bnet conser ave:", np.mean(safety_profiles), " bnet conser var:", np.var(safety_profiles))
print("bnet uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))

############################################





model = models.FCNet(model_param, mean, std, mean_label, std_label, device, bn=False).to(device)
model.load_state_dict(torch.load("./log/old/fcnet{:02d}/model_fc04.pth".format(1)))
model.eval()    

loss_no_noise, test1_no_noise, safety_no_noise, loc_xy_no_noise = test_general(model)

safety_profiles = []
loss_profiles = []
vari_profiles = []
time_profiles = []
loc_profiles = []
import time
model.add_noise = True
for i in range(100):
    start = time.time()
    loss, test1, safety, loc_xy = test_general(model)
    end = time.time()
    
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
    gt = test_labels_unnorm[:,0:2]
    np.save('./log/ctrl_gt.npy', gt)
    np.save('./log/ctrl_fc.npy', vari_np)
    np.save('./log/loc_fc.npy', loc_profiles.cpu().numpy())
    

print("FC time ave:", np.mean(time_profiles), " FC time var:", np.var(time_profiles))
print("FC loss ave:", np.mean(loss_profiles), " FC loss var:", np.var(loss_profiles))
print("FC Safety:", np.min(safety_profiles))
print("FC conser ave:", np.mean(safety_profiles), " FC conser var:", np.var(safety_profiles))
print("FC uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))
# import pdb; pdb.set_trace()


model = models.DFBNet(model_param, mean, std, mean_label, std_label, device, bn=False).to(device)
model.load_state_dict(torch.load("./log/old/fcnet{:02d}/model_fc04.pth".format(1)))
model.eval()    

loss_no_noise, test1_no_noise, safety_no_noise, loc_xy_no_noise = test_general(model)

safety_profiles = []
loss_profiles = []
vari_profiles = []
time_profiles = []
loc_profiles = []
model.add_noise = True
for i in range(100):
    start = time.time()
    loss, test1, safety, loc_xy = test_general(model)
    end = time.time()
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

if True: #write_file:
    vari_np = vari_profiles.cpu().numpy()
    # np.save('./log/ctrl_dfb.npy', vari_np)
    np.save('./log/loc_dfb1.npy', loc_profiles.cpu().numpy())
    import pdb; pdb.set_trace()

print("DFB time ave:", np.mean(time_profiles), " DFB time var:", np.var(time_profiles))
print("DFB loss ave:", np.mean(loss_profiles), " DFB loss var:", np.var(loss_profiles))
print("DFB Safety:", np.min(safety_profiles))
print("DFB conser ave:", np.mean(safety_profiles), " DFB conser var:", np.var(safety_profiles))
print("DFB uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))




model = models.BarrierNet(model_param, mean, std, mean_label, std_label, device, bn=False, activation = 'relu').to(device)
model.load_state_dict(torch.load("./log/bnet{:02d}".format(1) + "/model_bn09.pth"))
model.eval()    

loss_no_noise, test1_no_noise, safety_no_noise, loc_xy_no_noise = test_general(model)

safety_profiles = []
loss_profiles = []
vari_profiles = []
time_profiles = []
loc_profiles = []
model.add_noise = True
for i in range(100):
    start = time.time()
    loss, test1, safety, loc_xy = test_general(model)
    end = time.time()
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
    np.save('./log/loc_bnet.npy', loc_profiles.cpu().numpy())

print("bnet time ave:", np.mean(time_profiles), " bnet time var:", np.var(time_profiles))
print("bnet loss ave:", np.mean(loss_profiles), " bnet loss var:", np.var(loss_profiles))
print("bnet Safety:", np.min(safety_profiles))
print("bnet conser ave:", np.mean(safety_profiles), " bnet conser var:", np.var(safety_profiles))
print("bnet uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))


model_list = []
for i in range(10):
    model = models.BarrierNet(model_param, mean, std, mean_label, std_label, device, bn=False, activation = 'relu').to(device)
    model.load_state_dict(torch.load("./log/bnet{:02d}".format(i+1) + "/model_bn09.pth"))
    model.eval() 
    model.add_noise = True
    model_list.append(model)

loss_list = []
for i in range(10):
    loss, test1, safety, loc_xy = test_general(model_list[i])
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
    loss, test1, safety, loc_xy = abnet_test(model_list, wt_list)
    end = time.time()
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
    np.save('./log/loc_preabnet-10.npy', loc_profiles.cpu().numpy())

print("pre-abnet-10 time ave:", np.mean(time_profiles), " time var:", np.var(time_profiles))
print("pre-abnet-10 loss ave:", np.mean(loss_profiles), " loss var:", np.var(loss_profiles))
print("pre-abnet-10 Safety:", np.min(safety_profiles))
print("pre-abnet-10 conser ave:", np.mean(safety_profiles), " conser var:", np.var(safety_profiles))
print("pre-abnet-10 uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))

model = models.ABNet(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 10).to(device)
model.load_state_dict(torch.load("./log/abnet{:02d}".format(1) + "/model_abn09.pth"))
model.eval()    

loss_no_noise, test1_no_noise, safety_no_noise, loc_xy_no_noise = test_general(model)

safety_profiles = []
loss_profiles = []
vari_profiles = []
time_profiles = []
loc_profiles = []
model.add_noise = True
for i in range(100):
    start = time.time()
    loss, test1, safety, loc_xy = test_general(model)
    end = time.time()
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
    np.save('./log/loc_abnet-10.npy', loc_profiles.cpu().numpy())

print("abnet-10 time ave:", np.mean(time_profiles), " time var:", np.var(time_profiles))
print("abnet-10 loss ave:", np.mean(loss_profiles), " loss var:", np.var(loss_profiles))
print("abnet-10 Safety:", np.min(safety_profiles))
print("abnet-10 conser ave:", np.mean(safety_profiles), " conser var:", np.var(safety_profiles))
print("abnet-10 uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))



model = models.ABNet(model_param, mean, std, mean_label, std_label, device, bn=False, heads = 100).to(device)
model.load_state_dict(torch.load("./log/abnet{:02d}".format(3) + "/model_abn04.pth"))
model.eval()    

loss_no_noise, test1_no_noise, safety_no_noise, loc_xy_no_noise = test_general(model)

safety_profiles = []
loss_profiles = []
vari_profiles = []
time_profiles = []
loc_profiles = []
model.add_noise = True
for i in range(100):
    start = time.time()
    loss, test1, safety, loc_xy = test_general(model)
    end = time.time()
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
    np.save('./log/loc_abnet-100.npy', loc_profiles.cpu().numpy())

print("abnet-100 time ave:", np.mean(time_profiles), " time var:", np.var(time_profiles))
print("abnet-100 loss ave:", np.mean(loss_profiles), " loss var:", np.var(loss_profiles))
print("abnet-100 Safety:", np.min(safety_profiles))
print("abnet-100 conser ave:", np.mean(safety_profiles), " conser var:", np.var(safety_profiles))
print("abnet-100 uncertain:", torch.mean(torch.std(vari_profiles, dim = 0), dim = 0))


# ctrl1 = np.array(ctrl1)    
# ctrl1.tofile('ctrlM1_r12_dfb.dat')

# ctrl1_6 = np.fromfile('ctrl1_r6.dat', dtype=np.float64)
# ctrl1_6 = ctrl1_6.tolist()