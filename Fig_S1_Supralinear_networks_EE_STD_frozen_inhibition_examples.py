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
pal = sns.color_palette("deep")

sns.set(style='ticks')

# simulation setup
dt = 0.0001
T = int(9 / dt)

# neuronal parameters
tau_e, tau_i = 0.020, 0.010
alpha_e, alpha_i = 2, 2

# short-term depression
u_d, tau_x = 1, 0.20

# network connectivity
Jee = 1.5
Jie = 1.0
Jei = 2.0
Jii = 1.0

g_i = 5

l_g_e = [6.65, 20, 100] # [6.5, 20, 85, 90, 100]
l_r_e_total, l_r_i_total = [], []

for g_e in l_g_e:

    r_e, r_i = 0, 0
    z_e, z_i = 0, 0
    x = 1

    l_r_e, l_r_i, l_x = [], [], []
    Jacobian_mat = np.zeros((2, 2)) * np.nan

    for i in range(T):

        g_e = g_e * (g_e > 0)
        g_i = g_i * (g_i > 0)

        if 42000 <= i < 42001:
            r_e = r_e + 0.01
        else:
            pass

        # SSN part
        z_e = Jee * x * r_e - Jei * r_i + g_e
        z_i = Jie * r_e - Jii * r_i + g_i

        z_e = z_e * (z_e > 0)
        z_i = z_i * (z_i > 0)

        r_e = r_e + (-r_e + np.power(z_e, alpha_e)) / tau_e * dt
        if 40000 < i:
            pass
        else:
            r_i = r_i + (-r_i + np.power(z_i, alpha_i)) / tau_i * dt

        # if i == 40000:
        #     print("")
        #     print(g_e)
        #     Jacobian_mat[0, 0] = 1.0 / tau_e * (l_x[39999] * Jee * alpha_e * np.power(l_r_e[39999], (alpha_e - 1.0) / alpha_e) - 1)
        #     Jacobian_mat[0, 1] = 1.0 / tau_e * Jee * alpha_e * np.power(l_r_e[39999], (2 * alpha_e - 1.0) / alpha_e)
        #     Jacobian_mat[1, 0] = - u_d * l_x[39999]
        #     Jacobian_mat[1, 1] = -1.0 / tau_x - u_d * l_r_e[39999]
        #     lambda_1 = np.linalg.eig(Jacobian_mat)[0][0]
        #     lambda_2 = np.linalg.eig(Jacobian_mat)[0][1]
        #     print(lambda_1)
        #     print(lambda_2)

        r_e = r_e * (r_e > 0)
        r_i = r_i * (r_i > 0)

        x = x + ((1 - x) / tau_x - u_d * x * r_e) * dt
        x = np.clip(x, 0, 1)

        l_r_e.append(r_e)
        l_r_i.append(r_i)
        l_x.append(x)

    l_r_e = np.asarray(l_r_e)
    l_r_i = np.asarray(l_r_i)
    l_x = np.asarray(l_x)

    print("")
    print(g_e)
    print(l_r_e[30000])

    plt.figure(figsize=(figure_len, figure_width))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)
    plt.tick_params(width=line_width, length=tick_len)
    if g_e == 90 or g_e == 100:
        plt.yscale('linear')
    else:
        plt.yscale('symlog', linthreshy=1)

    plt.plot(l_r_e/l_r_e[30000], color='blue', linewidth=plot_line_width)
    plt.plot(l_r_i/l_r_i[30000], color='red', linewidth=plot_line_width)

    plt.xticks(np.arange(30000, 90000 + 5000, 20000), np.arange(0, 6 + 0.5, 2), fontsize=font_size_1, **hfont)
    if g_e == 6.65:
        plt.yticks([0, 1, 10], fontsize=font_size_1, **hfont)
    elif g_e == 20:
        plt.yticks([0, 1, 10, 100], fontsize=font_size_1, **hfont)
    elif g_e == 85:
        plt.yticks([0, 1, 10], fontsize=font_size_1, **hfont)
    else:
        plt.yticks([0.999, 1.0, 1.001], fontsize=font_size_1, **hfont)
    plt.xlabel('Time (s)', fontsize=font_size_1, **hfont)
    plt.ylabel('Normalized firing rate', fontsize=font_size_1, **hfont)

    plt.xlim([30000, 90000])
    if g_e == 6.65:
        plt.ylim([0, 10])
        plt.vlines(x=42001, ymin=0, ymax=10, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
    elif g_e == 20:
        plt.ylim([0, 100])
        plt.vlines(x=42001, ymin=0, ymax=100, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
    elif g_e == 85:
        plt.ylim([0, 10])
        plt.vlines(x=42001, ymin=0, ymax=10, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)
    else:
        plt.ylim([0.999, 1.001])
        plt.vlines(x=42001, ymin=0.999, ymax=1.001, colors='k', linestyles=[(0, (6, 6, 6, 6))], linewidth=line_width)

    plt.legend(['Exc', 'Inh'], prop={"family": "Arial", 'size': font_size_1}, loc='upper right')
    plt.savefig('paper_figures/png/Fig_1S_Supralinear_network_EE_STD_frozen_inhibition_examples_g_e_' + str(g_e) + '.png')
    plt.savefig('paper_figures/pdf/Fig_1S_Supralinear_network_EE_STD_frozen_inhibition_examples_g_e_' + str(g_e) + '.pdf')
