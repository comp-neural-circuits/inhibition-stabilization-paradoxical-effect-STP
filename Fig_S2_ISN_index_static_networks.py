import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.patches as mpatches
import scipy.io as sio

# plotting configuration
ratio = 1.5
figure_len, figure_width = 15*ratio, 12*ratio
font_size_1, font_size_2 = 36*ratio, 36*ratio
legend_size = 18*ratio
line_width, tick_len = 3*ratio, 12*ratio
marker_size = 15*ratio
marker_edge_width = 3 * ratio
plot_line_width = 5*ratio
hfont = {'fontname': 'Arial'}

sns.set(style='ticks')

tau_e = 0.020
l_Jee = np.arange(0, 3+0.001, 0.001)

x_max = 150
re = np.arange(0, x_max+0.001, 0.01)
alpha_e, alpha_i = 1, 1
ISN_mat_linear = np.zeros((len(l_Jee), len(re))) * np.nan

for Jee_idx in range(len(l_Jee)):
    Jee = l_Jee[Jee_idx]
    ISN_mat_linear[Jee_idx, :] = (1/tau_e) * (Jee * alpha_e * np.power(re, (alpha_e-1)/alpha_e) - 1)

plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

g = sns.heatmap(ISN_mat_linear, cmap="bwr", vmin=-20, vmax=20)
g.set_facecolor('gray')
g.set_xticklabels(g.get_xticklabels(), rotation=0)
g.set_yticklabels(g.get_yticklabels(), rotation=0)

plt.xticks(np.arange(0, 15001, 5000), [0, 50, 100, 150], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(0, 3001, 1000), [0, 1, 2, 3], fontsize=font_size_1, **hfont)

plt.xlim([0, len(re)-1])
plt.ylim([0, len(l_Jee)-1])
plt.xlabel(r'$r_E$', fontsize=font_size_1, **hfont)
plt.ylabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=font_size_1)
cbar = ax.collections[0].colorbar
cbar.set_ticks([-20, -10, 0, 10, 20])
plt.savefig('paper_figures/png/Fig_S3_ISN_index_linear_networks_EI_IE_II_STD_STP_mat.png')


x_max = 2
re = np.arange(0, x_max+0.0001, 0.001)
alpha_e, alpha_i = 1, 1
ISN_mat_linear = np.zeros((len(l_Jee), len(re))) * np.nan

for Jee_idx in range(len(l_Jee)):
    Jee = l_Jee[Jee_idx]
    ISN_mat_linear[Jee_idx, :] = (1/tau_e) * (Jee * alpha_e * np.power(re, (alpha_e-1)/alpha_e) - 1)

plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

g = sns.heatmap(ISN_mat_linear, cmap="bwr", vmin=-20, vmax=20)
g.set_facecolor('gray')
g.set_xticklabels(g.get_xticklabels(), rotation=0)
g.set_yticklabels(g.get_yticklabels(), rotation=0)

plt.xticks(np.arange(0, 2001, 1000), [0, 1, 2], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(0, 3001, 1000), [0, 1, 2, 3], fontsize=font_size_1, **hfont)

plt.xlim([0, len(re)-1])
plt.ylim([0, len(l_Jee)-1])
plt.xlabel(r'$r_E$', fontsize=font_size_1, **hfont)
plt.ylabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=font_size_1)
cbar = ax.collections[0].colorbar
cbar.set_ticks([-20, -10, 0, 10, 20])
plt.savefig('paper_figures/png/Fig_S3_ISN_index_linear_networks_EI_IE_II_STD_STP_mat_zoomin.png')


x_max = 150
re = np.arange(0, x_max+0.001, 0.01)
alpha_e, alpha_i = 2, 2
ISN_mat_supralinear = np.zeros((len(l_Jee), len(re))) * np.nan

for Jee_idx in range(len(l_Jee)):
    Jee = l_Jee[Jee_idx]
    ISN_mat_supralinear[Jee_idx, :] = (1/tau_e) * (Jee * alpha_e * np.power(re, (alpha_e-1)/alpha_e) - 1)

plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

g = sns.heatmap(ISN_mat_supralinear, cmap="bwr", vmin=-20, vmax=20)
g.set_facecolor('gray')
g.set_xticklabels(g.get_xticklabels(), rotation=0)
g.set_yticklabels(g.get_yticklabels(), rotation=0)

plt.xticks(np.arange(0, 15001, 5000), [0, 50, 100, 150], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(0, 3001, 1000), [0, 1, 2, 3], fontsize=font_size_1, **hfont)

plt.xlim([0, len(re)-1])
plt.ylim([0, len(l_Jee)-1])
plt.xlabel(r'$r_E$', fontsize=font_size_1, **hfont)
plt.ylabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=font_size_1)
cbar = ax.collections[0].colorbar
cbar.set_ticks([-20, -10, 0, 10, 20])
plt.savefig('paper_figures/png/Fig_S3_ISN_index_supralinear_networks_EI_IE_II_STD_STP_mat.png')


x_max = 2
re = np.arange(0, x_max+0.0001, 0.001)
alpha_e, alpha_i = 2, 2
ISN_mat_supralinear = np.zeros((len(l_Jee), len(re))) * np.nan

for Jee_idx in range(len(l_Jee)):
    Jee = l_Jee[Jee_idx]
    ISN_mat_supralinear[Jee_idx, :] = (1/tau_e) * (Jee * alpha_e * np.power(re, (alpha_e-1)/alpha_e) - 1)

plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

g = sns.heatmap(ISN_mat_supralinear, cmap="bwr", vmin=-20, vmax=20)
g.set_facecolor('gray')
g.set_xticklabels(g.get_xticklabels(), rotation=0)
g.set_yticklabels(g.get_yticklabels(), rotation=0)

plt.xticks(np.arange(0, 2001, 1000), [0, 1, 2], fontsize=font_size_1, **hfont)
plt.yticks(np.arange(0, 3001, 1000), [0, 1, 2, 3], fontsize=font_size_1, **hfont)

plt.xlim([0, len(re)-1])
plt.ylim([0, len(l_Jee)-1])
plt.xlabel(r'$r_E$', fontsize=font_size_1, **hfont)
plt.ylabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=font_size_1)
cbar = ax.collections[0].colorbar
cbar.set_ticks([-20, -10, 0, 10, 20])
plt.savefig('paper_figures/png/Fig_S3_ISN_index_supralinear_networks_EI_IE_II_STD_STP_mat_zoomin.png')