import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')


temp = np.fromfile('./plot/p1_increase.dat', dtype=np.float64)
temp = temp.tolist()
p1_increase = temp
temp = np.fromfile('./plot/p2_increase.dat', dtype=np.float64)
temp = temp.tolist()
p2_increase = temp

temp = np.fromfile('./plot/ref_a_increase.dat', dtype=np.float64)
temp = temp.tolist()
ref_a_increase = np.array(temp)
temp = np.fromfile('./plot/ref_omega_incease.dat', dtype=np.float64)
temp = temp.tolist()
ref_omega_increase = np.array(temp)
temp = np.fromfile('./plot/a_increase.dat', dtype=np.float64)
temp = temp.tolist()
a_increase = np.array(temp)
temp = np.fromfile('./plot/omega_incease.dat', dtype=np.float64)
temp = temp.tolist()
omega_increase = np.array(temp)


temp = np.fromfile('./plot/p1_decrease.dat', dtype=np.float64)
temp = temp.tolist()
p1_decrease = temp
temp = np.fromfile('./plot/p2_decrease.dat', dtype=np.float64)
temp = temp.tolist()
p2_decrease = temp

temp = np.fromfile('./plot/ref_a_decrease.dat', dtype=np.float64)
temp = temp.tolist()
ref_a_decrease = np.array(temp)
temp = np.fromfile('./plot/ref_omega_decease.dat', dtype=np.float64)
temp = temp.tolist()
ref_omega_decrease = np.array(temp)
temp = np.fromfile('./plot/a_decrease.dat', dtype=np.float64)
temp = temp.tolist()
a_decrease = np.array(temp)
temp = np.fromfile('./plot/omega_decease.dat', dtype=np.float64)
temp = temp.tolist()
omega_decrease = np.array(temp)

t = np.linspace(0,10,99)

plt.figure(1)
fig, ax = plt.subplots()

plt.plot([1,1,8,8,1], [-4.2,3.3,3.3,-4.2,-4.2], linestyle=':', color = 'black')
# plt.plot([8,8], [-4,3], linestyle=':', color = 'black')

plt.plot(t, p1_increase, color = 'red', label = '$p_1(z)$', linewidth=3.0)
plt.plot(t, p2_increase, linestyle='--', color = 'red', label = '$p_2(z)$', linewidth=3.0)
plt.plot(t, a_increase - ref_a_increase, color = 'b', label = '$a - a_{ref}$', linewidth=3.0)
plt.plot(t, omega_increase - ref_omega_increase, linestyle='--', color = 'b', label = '$\omega - \omega_{ref}$', linewidth=3.0)

# ax.set_aspect('equal')
# plt.xlim([-25, 15])
# plt.ylim([0, 40])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(loc ='lower right', prop={'size': 20}) #
# plt.ylabel('Penalties and errors',fontsize=14)
plt.xlabel('$t (s)$',fontsize=20)
plt.tight_layout()
plt.savefig('./plot/penalty_increase.pdf')

t = np.linspace(0,10,99)
plt.figure(2)
fig, ax = plt.subplots()
plt.plot([2.4,2.4,5,5,2.4], [-1.2,3.3,3.3,-1.2,-1.2], linestyle=':', color = 'black')

plt.plot(t, p1_decrease[20:119], color = 'red', label = '$p_1(z)$', linewidth=3.0)
plt.plot(t, p2_decrease[20:119], linestyle='--', color = 'red', label = '$p_2(z)$', linewidth=3.0)
plt.plot(t, a_decrease[20:119] - ref_a_decrease[20:119], color = 'b', label = '$a - a_{ref}$', linewidth=3.0)
plt.plot(t, omega_decrease[20:119] - ref_omega_decrease[20:119], linestyle='--',  color = 'b', label = '$\omega - \omega_{ref}$', linewidth=3.0)

# ax.set_aspect('equal')
# plt.xlim([-25, 15])
# plt.ylim([0, 40])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(loc ='lower right', prop={'size': 20})
# plt.ylabel('Penalties and errors',fontsize=14)
plt.xlabel('$t (s)$',fontsize=20)
plt.tight_layout()
plt.savefig('./plot/penalty_decrease.pdf')