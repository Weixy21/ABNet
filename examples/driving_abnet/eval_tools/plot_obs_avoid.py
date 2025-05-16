
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.io as sio
sns.set_style('whitegrid')

#ffmpeg -framerate 30 -pattern_type glob -i './png/*.png' -c:v libx264 -pix_fmt yuv420p ./out.mp4

ego_x_bn = []
ego_y_bn = []
ego_x_wbn = []
ego_y_wbn = []
for i in range(10):
    if i == 0:
        id = ''
    else:
        id = i
        id = format(id, '01d')
    temp = np.fromfile('./temp_data/ego_x_bn'+ id + '.dat', dtype=np.float64)
    temp = temp.tolist()
    ego_x_bn.append(temp)

    temp = np.fromfile('./temp_data/ego_y_bn'+ id + '.dat', dtype=np.float64)
    temp = temp.tolist()
    ego_y_bn.append(temp)

    temp = np.fromfile('./temp_data/ego_x_wbn'+ id + '.dat', dtype=np.float64)
    temp = temp.tolist()
    ego_x_wbn.append(temp)

    temp = np.fromfile('./temp_data/ego_y_wbn'+ id + '.dat', dtype=np.float64)
    temp = temp.tolist()
    ego_y_wbn.append(temp)

obs_x, obs_y = [], []
temp = np.fromfile('./temp_data/obs_x.dat', dtype=np.float64)
temp = temp.tolist()
obs_x.append(temp)

temp = np.fromfile('./temp_data/obs_y.dat', dtype=np.float64)
temp = temp.tolist()
obs_y.append(temp)


data = sio.loadmat('temp_data/mpc_data.mat')
mpc_data = np.float32(data['mpc_data'])

obs_xy = np.array([obs_x[0][0], obs_y[0][0]])


loc_abnet = np.load('temp_data/abnet2_loc.npy')
loc_abnet_att = np.load('temp_data/abnet_att2_loc.npy')
loc_abnet_sc = np.load('temp_data/abnet_sc2_loc.npy')
loc_dfb = np.load('temp_data/dfb3_loc.npy')
loc_bnet = np.load('temp_data/bnet2_loc.npy')
loc_bnet_up = np.load('temp_data/bnet-up_loc.npy')
loc_e2e = np.load('temp_data/e2e2_loc.npy')

control_abnet = np.load('temp_data/abnet_control.npy')
control_abnet_att = np.load('temp_data/abnet_att_control.npy')
control_bnet = np.load('temp_data/bnet_control.npy')
control_bnet_up = np.load('temp_data/bnet-up_control.npy')
control_dfb = np.load('temp_data/dfb2_control.npy')
control_e2e = np.load('temp_data/e2e_control.npy')


plt.figure(1)
fig, ax = plt.subplots()

obs = np.array([[-1, -2.5], [-1.0, +2.5], [+1.0, +2.5], [+1.0, -2.5], [-1.0, -2.5]])
# plt.plot(obs[:,0] + obs_x[0], obs[:,1] + obs_y[0], color = 'black')

plt.plot([-2.5,-2.5],[0,40], color = 'black')

# init = np.zeros([10,2])
# plt.plot(ego_x_bn[0], ego_y_bn[0], color = 'blue', label = 'BNet')
# init[0,:] = np.array([ego_x_bn[0][0], ego_y_bn[0][0]])
# for i in range(9):
#     plt.plot(ego_x_bn[i+1], ego_y_bn[i+1], color = 'blue')
#     init[i+1,:] = np.array([ego_x_bn[i+1][0], ego_y_bn[i+1][0]])




plt.plot(ego_x_wbn[0], ego_y_wbn[0], linestyle='--', color = 'green', label = 'V-E2E')
for i in range(9):
    plt.plot(ego_x_wbn[i+1], ego_y_wbn[i+1], linestyle='--', color = 'green')

# i, ct = 0,0
# plt.plot(loc_e2e[33,:,0], loc_e2e[33,:,1], linestyle='--', color = 'green', label = 'E2E-learning')
# plt.plot(loc_e2e[47,:,0], loc_e2e[47,:,1], linestyle='--', color = 'green')
# plt.plot(loc_e2e[37,:,0], loc_e2e[37,:,1], linestyle='--', color = 'green')
# # plt.plot(loc_e2e[24,:,0], loc_e2e[24,:,1], linestyle='--', color = 'green')
# # plt.plot(loc_e2e[13,:,0], loc_e2e[13,:,1], linestyle='--', color = 'green')
# # plt.plot(loc_e2e[14,:,0], loc_e2e[14,:,1], linestyle='--', color = 'green')
# while(True):
#     i += 1
#     # if np.max(loc_e2e[i,:55,0]) > 1:
#     #     continue
#     plt.plot(loc_e2e[i,:,0], loc_e2e[i,:,1], linestyle='--', color = 'green')
#     ct += 1
#     if ct > 7:
#         break


plt.plot(mpc_data[0:45,0], mpc_data[0:45, 1], linestyle='--', color = 'red', label = 'MPC')
for i in range(9):
    plt.plot(mpc_data[((i+1)*81+0):((i+1)*81 + 45),0], mpc_data[((i+1)*81+0):((i+1)*81 + 45),1], linestyle='--', color = 'red')

i, ct = 0, 0
plt.plot(loc_abnet[18,:,0], loc_abnet[18,:,1], linestyle='-', color = 'cyan', label = 'ABNet')
while(True):
    i += 1
    if np.max(loc_abnet[i,:55,0]) > 5:
        continue
    plt.plot(loc_abnet[i,:,0], loc_abnet[i,:,1], linestyle='-', color = 'cyan')
    ct += 1
    if ct > 9:
        break

i, ct = 0, 0
plt.plot(loc_abnet_att[18,:,0], loc_abnet_att[18,:,1], linestyle='-', color = 'magenta', label = 'ABNet-att')
while(True):
    i += 1
    if np.max(loc_abnet_att[i,:55,0]) > 5:
        continue
    plt.plot(loc_abnet_att[i,:,0], loc_abnet_att[i,:,1], linestyle='-', color = 'magenta')
    ct += 1
    if ct > 9:
        break


i, ct = 0, 0
plt.plot(loc_abnet_sc[18,:,0], loc_abnet_sc[18,:,1], linestyle=':', color = 'grey', label = 'ABNet-sc')
while(True):
    i += 1
    if np.max(loc_abnet_sc[i,:55,0]) > 5:
        continue
    plt.plot(loc_abnet_sc[i,:,0], loc_abnet_sc[i,:,1], linestyle=':', color = 'grey')
    ct += 1
    if ct > 9:
        break




i, ct = 0, 0
plt.plot(loc_bnet_up[18,:,0], loc_bnet_up[18,:,1], linestyle='--', color = 'purple', label = 'BNet-UP')
while(True):
    i += 1
    if np.max(loc_bnet_up[i,:55,0]) > 1 or i == 12:
        continue
    plt.plot(loc_bnet_up[i,:,0], loc_bnet_up[i,:,1], linestyle='--', color = 'purple')
    ct += 1
    if ct > 9:
        break

i, ct = 0, 0
plt.plot(loc_bnet[18,:,0], loc_bnet[18,:,1], linestyle='-', color = 'blue', label = 'BNet')
while(True):
    i += 1
    if np.max(loc_bnet[i,:55,0]) > 1 or i == 12:
        continue
    plt.plot(loc_bnet[i,:,0], loc_bnet[i,:,1], linestyle='-', color = 'blue')
    ct += 1
    if ct > 9:
        break

# i, ct = 0, 0
# plt.plot(loc_dfb[18,:,0], loc_dfb[18,:,1], linestyle='--', color = 'purple', label = 'dfb')
# while(True):
#     i += 1
#     if np.max(loc_dfb[i,:55,0]) > 1:
#         continue
#     plt.plot(loc_dfb[i,:,0], loc_dfb[i,:,1], linestyle='--', color = 'purple')
#     ct += 1
#     if ct > 9:
#         break

plt.plot([2.5,2.5],[0,40], color = 'black', label = 'Road boundaries')

ax.fill(obs[:,0] + obs_x[0], obs[:,1] + obs_y[0],"m",label = 'Obstacle')

ax.set_aspect('equal')
plt.xlim([-26, 15])
plt.ylim([0, 40])
ax.grid(False)

plt.legend(loc ='upper left', prop={'size': 12})
plt.ylabel('$y (m)$',fontsize=14)
plt.xlabel('$x (m)$',fontsize=14)
plt.tight_layout()
plt.savefig('./temp_data/obs_avoid_all.pdf')





############ reload again
loc_abnet = np.load('temp_data/abnet_loc.npy')
loc_abnet_att = np.load('temp_data/abnet_att_loc.npy')
loc_bnet = np.load('temp_data/bnet_loc.npy')

control_abnet = np.load('temp_data/abnet_control.npy')
control_abnet_att = np.load('temp_data/abnet_att_control.npy')
control_bnet = np.load('temp_data/bnet_control.npy')

control_abnet_new = []
control_abnet_att_new = []
control_bnet_new = []
control_bnet_up_new = []
for i in range(100):
    if np.max(loc_bnet[i,:55,0]) <= 2:
        control_bnet_new.append(control_bnet[i:i+1,:,:])
    if np.max(loc_abnet[i,:55,0]) <= 2:
        control_abnet_new.append(control_abnet[i:i+1,:,:])
    if np.max(loc_abnet_att[i,:55,0]) <= 2:
        control_abnet_att_new.append(control_abnet_att[i:i+1,:,:])
for i in range(50):
    if np.max(loc_bnet_up[i,:55,0]) <= 2:
        control_bnet_up_new.append(control_bnet_up[i:i+1,:,:])

control_abnet = np.concatenate(control_abnet_new, axis = 0)
control_bnet = np.concatenate(control_bnet_new, axis = 0)
control_abnet_att = np.concatenate(control_abnet_att_new, axis = 0)
control_bnet_up = np.concatenate(control_bnet_up_new, axis = 0)


max_ctrl_abnet = np.max(control_abnet, axis=0)
min_ctrl_abnet = np.min(control_abnet, axis=0)

max_ctrl_abnet_att = np.max(control_abnet_att, axis=0)
min_ctrl_abnet_att = np.min(control_abnet_att, axis=0)

max_ctrl_bnet = np.max(control_bnet, axis=0)
min_ctrl_bnet = np.min(control_bnet, axis=0)

max_ctrl_bnet_up = np.max(control_bnet_up, axis=0)
min_ctrl_bnet_up = np.min(control_bnet_up, axis=0)

tr = np.arange(0, 99*0.1, 0.1) 
plt.figure(2)
fig, ax = plt.subplots()

# plt.plot(tr, ctrl_gt[:,0], color = 'black', label = 'Ground truth')
plt.plot(tr, control_bnet[3,:,0], color = 'red', label = 'BNet')  #2
plt.fill_between(tr, max_ctrl_bnet[:,0], min_ctrl_bnet[:,0], color = 'red', alpha = 0.2, label = 'BNet distribution')
plt.plot(tr, control_bnet_up[3,:,0], color = 'purple', label = 'BNet-UP')  #2
plt.fill_between(tr, max_ctrl_bnet_up[:,0], min_ctrl_bnet_up[:,0], color = 'purple', alpha = 0.2, label = 'BNet-UP distribution')
plt.plot(tr, control_abnet[3,:,0], color = 'blue', label = 'ABNet')
plt.fill_between(tr, max_ctrl_abnet[:,0], min_ctrl_abnet[:,0], color = 'blue', alpha = 0.2, label = 'ABNet distribution')
plt.plot(tr, control_abnet_att[3,:,0], color = 'green', label = 'ABNet-att')   ###2
plt.fill_between(tr, max_ctrl_abnet_att[:,0], min_ctrl_abnet_att[:,0], color = 'green', alpha = 0.2, label = 'ABNet-att distribution')

plt.legend(loc ='lower right', prop={'size': 9})
plt.ylabel('Acceleration $u_1/(m/s^2)$',fontsize=14)
plt.xlabel('time$/s$',fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.set_rasterized(True)

plt.savefig('./temp_data/test_u1.pdf')    
# plt.show()

plt.figure(3)
fig, ax = plt.subplots()
# plt.plot(tr, ctrl_gt[:,1], color = 'black', label = 'Ground truth')
plt.plot(tr, control_bnet[3,:,1], color = 'red', label = 'BNet')
plt.fill_between(tr, max_ctrl_bnet[:,1], min_ctrl_bnet[:,1], color = 'red', alpha = 0.2, label = 'BNet distribution')
plt.plot(tr, control_bnet_up[3,:,1], color = 'magenta', label = 'BNet-UP')  #2
plt.fill_between(tr, max_ctrl_bnet_up[:,1], min_ctrl_bnet_up[:,1], color = 'magenta', alpha = 0.2, label = 'BNet-UP distribution')
plt.plot(tr, control_abnet[3,:,1], color = 'blue', label = 'ABNet')
plt.fill_between(tr, max_ctrl_abnet[:,1], min_ctrl_abnet[:,1], color = 'blue', alpha = 0.2, label = 'ABNet distribution')
plt.plot(tr, control_abnet_att[3,:,1], color = 'green', label = 'ABNet-att')
plt.fill_between(tr, max_ctrl_abnet_att[:,1], min_ctrl_abnet_att[:,1], color = 'green', alpha = 0.2, label = 'ABNet-att distribution')
plt.legend(loc ='lower center',prop={'size': 9})  #, frameon=False
plt.ylabel('Steering rate $u_2/(rad/s)$',fontsize=14)
plt.xlabel('time$/s$',fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.set_rasterized(True)

plt.savefig('./temp_data/test_u2.pdf') 


tr = [1, 10, 20, 30, 50, 70, 100]
crash = [0.03, 0, 0, 0, 0, 0, 0]
passr = [0.33, 1, 1, 1, 1, 1, 1]
out = [0, 0.31, 0.50, 0.46, 0.43, 0.43, 0.73]
u1 = np.array([0.72419802, 0.16827857, 0.13018149, 0.13453956, 0.13012366, 0.12174663, 0.14660542])
u2 = np.array([0.38520539, 0.31644732, 0.25612661, 0.25005959, 0.24721765, 0.23500988, 0.23199912])

plt.figure(4)
fig, ax = plt.subplots()
ax.plot(tr, crash, color = 'cyan', label = 'Crash rate', linewidth = 3)
ax.plot(tr, passr, color = 'blue', label = 'Pass rate', linewidth = 3)  #2
ax.plot(tr, crash, 'o', color = 'cyan',  linewidth = 3)
ax.plot(tr, passr, 'o', color = 'blue', linewidth = 3)
ax.plot([-3,-2], [-1, -1], color = 'red', label = 'Acceleration $u_1$x10 uncertainty', linewidth = 3)
ax.plot([-3,-2], [-1, -1], color = 'green', label = 'Steering rate $u_2$x10 uncertainty', linewidth = 3)
ax.set_xlim([-4,104])
ax.set_ylim([-0.05, 1.05])
# ax.plot(tr, out, color = 'blue', label = 'Out lane rate', linewidth = 3)
ax2 = ax.twinx()
ax2.plot(tr, 10*u1, color = 'red', label = 'Acceleration $10u_1$ uncertainty', linewidth = 3)
# ax2.set_yticks([])
# ax3 = ax.twinx()
ax2.plot(tr, 10*u2, color = 'green', label = 'Steering rate $10u_2$ uncertainty', linewidth = 3)

ax2.plot(tr, 10*u1, 'o', color = 'red',  linewidth = 3)
ax2.plot(tr, 10*u2, 'o', color = 'green',  linewidth = 3)
#ax3.set_yticks([])
ax.legend(loc ='upper right',prop={'size': 12})  #, frameon=False
ax2.set_ylim([1,4])
ax.set_xlabel('$h$: number of BarrierNet heads with scalable training',fontsize=14)
ax.set_ylabel('Rates',fontsize=14)
ax2.set_ylabel('Uncertainties',fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.set_rasterized(True)

plt.savefig('./temp_data/scale.pdf') 




