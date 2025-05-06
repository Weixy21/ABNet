import torch
import torch.nn as nn
from my_classes import Dataset_list as Dataset
import scipy.io as sio
import models
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
plt.style.use('bmh')
from scipy.integrate import odeint
import os
import time
#export PYTHONNOUSERSITE=True   for python
#export PYOPENGL_PLATFORM=egl   for Vista (cannot connect to "%s"' % name)

root = "./log/closed-form/"

#choose abnet, barriernet or not
barriernet = 0
abnet = 1 
exp = 0

#dynamics
def dynamics(y,t):
    dxdt = y(3)*np.cos(y(2))
    dydt = y(3)*np.sin(y(2))
    dttdt = y(4) #u1
    dvdt = y(5)  #u2
    return [dxdt,dydt,dttdt,dvdt]

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using {} device".format(device))
torch.backends.cudnn.benchmark = True


# Datasets
train_data = sio.loadmat('data/dataM_train.mat') 
train_data = train_data['data']
valid_data = sio.loadmat('data/dataM_valid.mat') 
valid_data = valid_data['data']
test_data = sio.loadmat('data/dataM_test.mat') 
test_data = test_data['data']

train0 = np.double(train_data[:,0:5])  # px, py, theta, v, dst_y, //dst_x is fixed
train_labels = np.reshape(np.double(train_data[:,5:7]), (len(train_data),2)) #theta_derivative, acc
valid0 = np.double(valid_data[:,0:5]) 
valid_labels = np.reshape(np.double(valid_data[:,5:7]), (len(valid_data),2))
test0 = np.double(test_data[:,0:5]) 
test_labels = np.reshape(np.double(test_data[:,5:7]), (len(test_data),2))
init = test0[0]

mean = np.mean(train0, axis = 0)
std= np.std(train0, axis = 0)

train0 = (train0 - mean)/std
valid0 = (valid0 - mean)/std
test0 = (test0 - mean)/std


# Parameters
params = {'batch_size': 64,    # 64
          'shuffle': True,
          'num_workers': 20}  # 20

# Generators
training_set = Dataset(train0, train_labels)
train_dataloader = torch.utils.data.DataLoader(training_set, **params)

valid_set = Dataset(valid0, valid_labels)
valid_dataloader = torch.utils.data.DataLoader(valid_set, **params)


# Initialize the model.
nFeatures, nHidden1, nHidden21, nHidden22, nCls = 5, 128, 32, 32, 2 
if barriernet == 1:
    model = models.BarrierNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
elif abnet == 1:
    model = models.ABNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False, heads = 10).to(device)
    model.use_cf = True  # closed form
else:
    model = models.FCNet(nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn=False).to(device)
print(model)


# Initialize the optimizer.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def train(dataloader, model, loss_fn, optimizer, losses, itr):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        if abnet == 1:
            pred = model(X, 1, itr)
        else:
            pred = model(X, 1)
        loss = loss_fn(pred, y)
        losses.append(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 25 == 0:  #25
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return losses

def test(dataloader, model, loss_fn, losses, itr):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            if abnet == 1:
                pred = model(X, 1, itr)
            else:
                pred = model(X, 1)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
    test_loss /= num_batches
    losses.append(test_loss)
    print(f"Test avg loss: {test_loss:>8f} \n")
    return losses

    
epochs = 20
train_losses, test_losses = [], []
start_time = time.time()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_losses = train(train_dataloader, model, loss_fn, optimizer, train_losses, t)
    test_losses = test(valid_dataloader, model, loss_fn, test_losses, t)
    if barriernet == 1:
        makedirs(root + 'bnet{:02d}'.format(exp))
        torch.save(model.state_dict(), root + "bnet{:02d}".format(exp) + "/model_bn{:02d}.pth".format(t))
    elif abnet == 1:
        makedirs(root + 'abnet{:02d}'.format(exp))
        torch.save(model.state_dict(), root + "abnet{:02d}".format(exp) + "/model_abn{:02d}.pth".format(t))
    else:
        torch.save(model.state_dict(), "model_fc.pth")
    print("Saved PyTorch Model State to model_{:02d}.pth".format(t))
print("Training Done!")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time/3600} hours")


model.eval()    
tr = []
ctrl1, ctrl2, ctrl1_real, ctrl2_real = [], [], [], []
t0 = 0


with torch.no_grad():
    for i in range(0,len(test0),1):
        x = Variable(torch.from_numpy(test0[i]), requires_grad=False)
        x = torch.reshape(x, (1,nFeatures))
        x = x.to(device)
        if abnet == 1:
            ctrl = model(x, 0, 0)
        else:
            ctrl = model(x, 0)
        
        if barriernet == 1:
            ctrl1.append(ctrl[0])
            ctrl2.append(ctrl[1])
        else:
            ctrl1.append(ctrl[0,0].item())
            ctrl2.append(ctrl[0,1].item())
        ctrl1_real.append(test_labels[i][0])
        ctrl2_real.append(test_labels[i][1])
        tr.append(t0)
        t0 = t0 + 0.1

print("Test done!")    


plt.figure(1)
plt.plot(tr, ctrl1_real, color = 'red', label = 'actual(optimal)')
plt.plot(tr, ctrl1, color = 'blue', label = 'implemented')
plt.legend()
plt.ylabel('Angular speed (control)')
plt.xlabel('time')
# plt.show()
if barriernet == 1:
    plt.savefig(root + 'bnet{:02d}/train_test_u1'.format(exp))
if abnet == 1:
    plt.savefig(root + 'abnet{:02d}/train_test_u1'.format(exp))      

plt.figure(2)
plt.plot(tr, ctrl2_real, color = 'red', label = 'actual(optimal)')
plt.plot(tr, ctrl2, color = 'blue', label = 'implemented')
plt.legend()
plt.ylabel('Acceleration (control)')
plt.xlabel('time')
if barriernet == 1:
    plt.savefig(root + 'bnet{:02d}/train_test_u2'.format(exp)) 
if abnet == 1:
    plt.savefig(root + 'abnet{:02d}/train_test_u2'.format(exp))   
# plt.show()

plt.figure(3)    
plt.plot(train_losses, color = 'green', label = 'train')
#plt.plot(test_losses, color = 'red', label = 'test')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('time')
plt.ylim(ymin=0.)
# plt.show()
if barriernet == 1:
    plt.savefig(root + 'bnet{:02d}/train_loss'.format(exp))
if abnet == 1:
    plt.savefig(root + 'abnet{:02d}/train_loss'.format(exp))
print("end")