import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
from matplotlib import patches
import matplotlib.patches as mpatches
import scipy.io as sio
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from sympy.solvers import solve
from sympy import Symbol
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import scipy.io as sio

session = WolframLanguageSession()

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
pal = sns.color_palette("deep")

b_trace = True
b_determinant = True
b_zoomin = True

# short-term depression
l_u_d = [0.2, 0.6, 1.0]
tau_x = 0.20
tau_e = 0.020

if b_zoomin:
    x_max = 2
    l_re = np.arange(0, x_max+0.0001, 0.001)
else:
    x_max = 150
    l_re = np.arange(0, x_max+0.001, 0.01)

alpha_e, alpha_i = 1, 1
trace_boundary_value = np.zeros((len(l_u_d), len(l_re))) * np.nan

if b_trace:
    for u_d_idx in range(len(l_u_d)):
        u_d = l_u_d[u_d_idx]
        for re_idx in range(len(l_re)):
            re = l_re[re_idx]

            string = """ {solve_for} /. Solve[{{ (1.0/{tau_e}) * (Jee * {alpha_e} * (({re})^(({alpha_e}-1)/{alpha_e}))/(1 + {u_d} * {re} * {tau_x}) - 1) - 1/{tau_x} - {u_d} * {re}==0}}, {solve_for}, Reals]""".format(
                tau_e=tau_e, tau_x=tau_x, u_d=u_d, re=re, alpha_e=alpha_e, alpha_i=alpha_i, solve_for='{Jee}')

            P = session.evaluate(string)
            if len(P) == 1:
                solution_string = str(P[0])
                solution_Jee = solution_string.split(",")[0]
                if solution_Jee != 'Global`Jee':
                    Jee_solution = float(solution_Jee[1:])
                    trace_boundary_value[u_d_idx, re_idx] = Jee_solution
                else:
                    pass


det_boundary_value = np.zeros((len(l_u_d), len(l_re))) * np.nan

if b_determinant:
    for u_d_idx in range(len(l_u_d)):
        u_d = l_u_d[u_d_idx]
        for re_idx in range(len(l_re)):
            re = l_re[re_idx]

            string = """ {solve_for} /. Solve[{{ - (1.0/{tau_e}) * (Jee * {alpha_e} * (({re})^(({alpha_e}-1)/{alpha_e}))/{tau_x} - (1 + {tau_x} * {u_d} * {re})/{tau_x} - (Jee * {alpha_e} * (({re})^((2 * {alpha_e}-1)/{alpha_e})) * {u_d})/(1 + {u_d} * {re} * {tau_x}))==0}}, {solve_for}, Reals]""".format(
                tau_e=tau_e, tau_x=tau_x, u_d=u_d, re=re, alpha_e=alpha_e, alpha_i=alpha_i, solve_for='{Jee}')

            P = session.evaluate(string)
            if len(P) == 1:
                solution_string = str(P[0])
                solution_Jee = solution_string.split(",")[0]
                if solution_Jee != 'Global`Jee':
                    Jee_solution = float(solution_Jee[1:])
                    det_boundary_value[u_d_idx, re_idx] = Jee_solution
                else:
                    pass


print("Linear networks")
print(trace_boundary_value[0, :])
print(det_boundary_value[0, :])
print(len(np.where(trace_boundary_value[0, :] > det_boundary_value[0, :])[0]))
print(len(np.where(trace_boundary_value[1, :] > det_boundary_value[1, :])[0]))
print(len(np.where(trace_boundary_value[2, :] > det_boundary_value[2, :])[0]))
print(len(np.where(trace_boundary_value[0, :] < det_boundary_value[0, :])[0]))
print(len(np.where(trace_boundary_value[1, :] < det_boundary_value[1, :])[0]))
print(len(np.where(trace_boundary_value[2, :] < det_boundary_value[2, :])[0]))

plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

if b_trace:
    plt.plot(trace_boundary_value[0, :], linewidth=plot_line_width)
    plt.plot(trace_boundary_value[1, :], linewidth=plot_line_width)
    plt.plot(trace_boundary_value[2, :], linewidth=plot_line_width)

if b_determinant:
    plt.plot(det_boundary_value[0, :], color=pal[0], linestyle='dashed', linewidth=plot_line_width)
    plt.plot(det_boundary_value[1, :], color=pal[1], linestyle='dashed', linewidth=plot_line_width)
    plt.plot(det_boundary_value[2, :], color=pal[2], linestyle='dashed', linewidth=plot_line_width)

if b_zoomin:
    plt.xticks(np.arange(0, 2001, 1000), [0, 1, 2], fontsize=font_size_1, **hfont)
else:
    plt.xticks(np.arange(0, 15001, 5000), [0, 50, 100, 150], fontsize=font_size_1, **hfont)

plt.yticks([0, 1, 2, 3], fontsize=font_size_1, **hfont)

plt.xlim([0, len(l_re)-1])
plt.ylim([0, 3])
plt.xlabel(r'$r_E$', fontsize=font_size_1, **hfont)
plt.ylabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
plt.legend([r'$u_d$: 0.2', r'$u_d$: 0.6', r'$u_d$: 1.0'], prop={"family": "Arial", 'size': font_size_1}, loc='best')
if b_trace and b_determinant:
    if b_zoomin:
        plt.savefig('paper_figures/png/Fig_1_Linear_networks_EE_STD_trace_det_boundary_zoomin.png')
        plt.savefig('paper_figures/pdf/Fig_1_Linear_networks_EE_STD_trace_det_boundary_zoomin.pdf')
    else:
        plt.savefig('paper_figures/png/Fig_1_Linear_networks_EE_STD_trace_det_boundary.png')
        plt.savefig('paper_figures/pdf/Fig_1_Linear_networks_EE_STD_trace_det_boundary.pdf')
elif b_trace:
    plt.savefig('paper_figures/png/Fig_1_Linear_networks_EE_STD_trace_boundary.png')
    plt.savefig('paper_figures/pdf/Fig_1_Linear_networks_EE_STD_trace_boundary.pdf')
else:
    plt.savefig('paper_figures/png/Fig_1_Linear_networks_EE_STD_det_boundary.png')
    plt.savefig('paper_figures/pdf/Fig_1_Linear_networks_EE_STD_det_boundary.pdf')


plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

ISN_boundary_value = np.zeros((len(l_u_d), len(l_re))) * np.nan
for u_d_idx in range(len(l_u_d)):
    for re_idx in range(len(l_re)):
        if det_boundary_value[u_d_idx, re_idx] < trace_boundary_value[u_d_idx, re_idx]:
            ISN_boundary_value[u_d_idx, re_idx] = det_boundary_value[u_d_idx, re_idx]
        else:
            ISN_boundary_value[u_d_idx, re_idx] = trace_boundary_value[u_d_idx, re_idx]

plt.plot(ISN_boundary_value[0, :], linewidth=plot_line_width)
plt.plot(ISN_boundary_value[1, :], linewidth=plot_line_width)
plt.plot(ISN_boundary_value[2, :], linewidth=plot_line_width)

if b_zoomin:
    plt.xticks(np.arange(0, 2001, 1000), [0, 1, 2], fontsize=font_size_1, **hfont)
else:
    plt.xticks(np.arange(0, 15001, 5000), [0, 50, 100, 150], fontsize=font_size_1, **hfont)
plt.yticks([0, 1, 2, 3], fontsize=font_size_1, **hfont)

plt.xlim([0, len(l_re)-1])
plt.ylim([0, 3])
plt.xlabel(r'$r_E$', fontsize=font_size_1, **hfont)
plt.ylabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
plt.legend([r'$u_d$: 0.2', r'$u_d$: 0.6', r'$u_d$: 1.0'], prop={"family": "Arial", 'size': font_size_1}, loc='best')
if b_zoomin:
    plt.savefig('paper_figures/png/Fig_1_Linear_networks_EE_STD_ISN_boundary_zoomin.png')
    plt.savefig('paper_figures/pdf/Fig_1_Linear_networks_EE_STD_ISN_boundary_zoomin.pdf')
else:
    plt.savefig('paper_figures/png/Fig_1_Linear_networks_EE_STD_ISN_boundary.png')
    plt.savefig('paper_figures/pdf/Fig_1_Linear_networks_EE_STD_ISN_boundary.pdf')


# short-term depression
l_u_d = [0.2, 0.6, 1.0]
tau_x = 0.20
tau_e = 0.020

if b_zoomin:
    x_max = 30
    l_re = np.arange(0, x_max+0.001, 0.005)
else:
    x_max = 150
    l_re = np.arange(0, x_max+0.001, 0.01)

alpha_e, alpha_i = 2, 2
trace_boundary_value = np.zeros((len(l_u_d), len(l_re))) * np.nan

if b_trace:
    for u_d_idx in range(len(l_u_d)):
        u_d = l_u_d[u_d_idx]
        for re_idx in range(len(l_re)):
            re = l_re[re_idx]

            string = """ {solve_for} /. Solve[{{ (1.0/{tau_e}) * (Jee * {alpha_e} * (({re})^(({alpha_e}-1)/{alpha_e}))/(1 + {u_d} * {re} * {tau_x}) - 1) - 1/{tau_x} - {u_d} * {re}==0}}, {solve_for}, Reals]""".format(
                tau_e=tau_e, tau_x=tau_x, u_d=u_d, re=re, alpha_e=alpha_e, alpha_i=alpha_i, solve_for='{Jee}')

            P = session.evaluate(string)
            if len(P) == 1:
                solution_string = str(P[0])
                solution_Jee = solution_string.split(",")[0]
                if solution_Jee != 'Global`Jee':
                    Jee_solution = float(solution_Jee[1:])
                    trace_boundary_value[u_d_idx, re_idx] = Jee_solution
                else:
                    pass


det_boundary_value = np.zeros((len(l_u_d), len(l_re))) * np.nan

if b_determinant:
    for u_d_idx in range(len(l_u_d)):
        u_d = l_u_d[u_d_idx]
        for re_idx in range(len(l_re)):
            re = l_re[re_idx]

            string = """ {solve_for} /. Solve[{{ - (1.0/{tau_e}) * (Jee * {alpha_e} * (({re})^(({alpha_e}-1)/{alpha_e}))/{tau_x} - (1 + {tau_x} * {u_d} * {re})/{tau_x} - (Jee * {alpha_e} * (({re})^((2 * {alpha_e}-1)/{alpha_e})) * {u_d})/(1 + {u_d} * {re} * {tau_x}))==0}}, {solve_for}, Reals]""".format(
                tau_e=tau_e, tau_x=tau_x, u_d=u_d, re=re, alpha_e=alpha_e, alpha_i=alpha_i, solve_for='{Jee}')

            P = session.evaluate(string)
            if len(P) == 1:
                solution_string = str(P[0])
                solution_Jee = solution_string.split(",")[0]
                if solution_Jee != 'Global`Jee':
                    Jee_solution = float(solution_Jee[1:])
                    det_boundary_value[u_d_idx, re_idx] = Jee_solution
                else:
                    pass

print("Supralinear networks")
print(len(np.where(trace_boundary_value[0, :] > det_boundary_value[0, :])[0]))
print(len(np.where(trace_boundary_value[1, :] > det_boundary_value[1, :])[0]))
print(len(np.where(trace_boundary_value[2, :] > det_boundary_value[2, :])[0]))
print(len(np.where(trace_boundary_value[0, :] < det_boundary_value[0, :])[0]))
print(len(np.where(trace_boundary_value[1, :] < det_boundary_value[1, :])[0]))
print(len(np.where(trace_boundary_value[2, :] < det_boundary_value[2, :])[0]))

plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

if b_trace:
    plt.plot(trace_boundary_value[0, :], linewidth=plot_line_width)
    plt.plot(trace_boundary_value[1, :], linewidth=plot_line_width)
    plt.plot(trace_boundary_value[2, :], linewidth=plot_line_width)

if b_determinant:
    plt.plot(det_boundary_value[0, :], color=pal[0], linestyle='dashed', linewidth=plot_line_width)
    plt.plot(det_boundary_value[1, :], color=pal[1], linestyle='dashed', linewidth=plot_line_width)
    plt.plot(det_boundary_value[2, :], color=pal[2], linestyle='dashed', linewidth=plot_line_width)

if b_zoomin:
    plt.xticks(np.arange(0, 2001, 1000), [0, 1, 2], fontsize=font_size_1, **hfont)
else:
    plt.xticks(np.arange(0, 15001, 5000), [0, 50, 100, 150], fontsize=font_size_1, **hfont)
plt.yticks([0, 1, 2, 3], fontsize=font_size_1, **hfont)

plt.xlim([0, len(l_re)-1])
plt.ylim([0, 3])
plt.xlabel(r'$r_E$', fontsize=font_size_1, **hfont)
plt.ylabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
plt.legend([r'$u_d$: 0.2', r'$u_d$: 0.6', r'$u_d$: 1.0'], prop={"family": "Arial", 'size': font_size_1}, loc='best')
if b_trace and b_determinant:
    if b_zoomin:
        plt.savefig('paper_figures/png/Fig_1_Supralinear_networks_EE_STD_trace_det_boundary_zoomin.png')
        plt.savefig('paper_figures/pdf/Fig_1_Supralinear_networks_EE_STD_trace_det_boundary_zoomin.pdf')
    else:
        plt.savefig('paper_figures/png/Fig_1_Supralinear_networks_EE_STD_trace_det_boundary.png')
        plt.savefig('paper_figures/pdf/Fig_1_Supralinear_networks_EE_STD_trace_det_boundary.pdf')

elif b_trace:
    plt.savefig('paper_figures/png/Fig_1_Supralinear_networks_EE_STD_trace_boundary.png')
    plt.savefig('paper_figures/pdf/Fig_1_Supralinear_networks_EE_STD_trace_boundary.pdf')
else:
    plt.savefig('paper_figures/png/Fig_1_Supralinear_networks_EE_STD_det_boundary.png')
    plt.savefig('paper_figures/pdf/Fig_1_Supralinear_networks_EE_STD_det_boundary.pdf')


plt.figure(figsize=(figure_len, figure_width))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(line_width)
plt.tick_params(width=line_width, length=tick_len)

ISN_boundary_value = np.zeros((len(l_u_d), len(l_re))) * np.nan
for u_d_idx in range(len(l_u_d)):
    for re_idx in range(len(l_re)):
        if det_boundary_value[u_d_idx, re_idx] < trace_boundary_value[u_d_idx, re_idx]:
            ISN_boundary_value[u_d_idx, re_idx] = det_boundary_value[u_d_idx, re_idx]
        else:
            ISN_boundary_value[u_d_idx, re_idx] = trace_boundary_value[u_d_idx, re_idx]

plt.plot(ISN_boundary_value[0, :], linewidth=plot_line_width)
plt.plot(ISN_boundary_value[1, :], linewidth=plot_line_width)
plt.plot(ISN_boundary_value[2, :], linewidth=plot_line_width)

if b_zoomin:
    plt.xticks(np.arange(0, 6001, 3000), [0, 15, 30], fontsize=font_size_1, **hfont)
else:
    plt.xticks(np.arange(0, 15001, 5000), [0, 50, 100, 150], fontsize=font_size_1, **hfont)
plt.yticks([0, 1, 2, 3], fontsize=font_size_1, **hfont)

plt.xlim([0, len(l_re)-1])
plt.ylim([0, 3])
plt.xlabel(r'$r_E$', fontsize=font_size_1, **hfont)
plt.ylabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
plt.legend([r'$u_d$: 0.2', r'$u_d$: 0.6', r'$u_d$: 1.0'], prop={"family": "Arial", 'size': font_size_1}, loc='best')
if b_zoomin:
    plt.savefig('paper_figures/png/Fig_1_Supralinear_networks_EE_STD_ISN_boundary_zoomin.png')
    plt.savefig('paper_figures/pdf/Fig_1_Supralinear_networks_EE_STD_ISN_boundary_zoomin.pdf')
else:
    plt.savefig('paper_figures/png/Fig_1_Supralinear_networks_EE_STD_ISN_boundary.png')
    plt.savefig('paper_figures/pdf/Fig_1_Supralinear_networks_EE_STD_ISN_boundary.pdf')