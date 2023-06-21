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

# short-term depression
l_u_d = [0.2, 0.6, 1.0]
tau_x = 0.20
tau_e = 0.020

x_max = 50
l_re = np.arange(0, x_max+0.001, 0.005)

alpha_e, alpha_i = 2, 2
trace_boundary_value = np.zeros((len(l_u_d), len(l_re))) * np.nan
turning_points_trace_re = []
turning_points_trace_Jee = []
turning_points_det_re = []
turning_points_det_Jee = []
turning_points_IS = []

# turning point part
for u_d_idx in range(len(l_u_d)):
    u_d = l_u_d[u_d_idx]
    turning_points_trace_re.append((-(2 * tau_e + tau_x) + np.sqrt(4 * np.power(alpha_e, 2) * np.power(tau_e, 2) + 4 * np.power(alpha_e, 2) * tau_e * tau_x + np.power(tau_x, 2)))/(2 * (alpha_e + 1) * tau_e * tau_x * u_d))
    turning_points_det_re.append((alpha_e - 1)/(tau_x * u_d * (1 + alpha_e)))

for i in range(len(turning_points_trace_re)):
    u_d = l_u_d[i]
    trace_re = turning_points_trace_re[i]
    turning_points_trace_Jee.append((tau_e/tau_x + 1 + 2 * tau_e * u_d * trace_re + tau_e * tau_x * np.power(u_d, 2) * np.power(trace_re, 2) + u_d * tau_x * trace_re)/(alpha_e * np.power(trace_re, (alpha_e-1)/alpha_e)))

for j in range(len(turning_points_det_re)):
    u_d = l_u_d[j]
    det_re = turning_points_det_re[j]
    turning_points_det_Jee.append(np.power(1 + tau_x * u_d * det_re, 2)/ (alpha_e * np.power(det_re, (alpha_e - 1) / alpha_e)))

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
plt.plot(turning_points_trace_re[0]*200, turning_points_trace_Jee[0], 'o', markerfacecolor=pal[0], markeredgecolor='none', markersize=marker_size*1.5)
plt.plot(turning_points_trace_re[1]*200, turning_points_trace_Jee[1], 'o', markerfacecolor=pal[1], markeredgecolor='none', markersize=marker_size*1.5)
plt.plot(turning_points_trace_re[2]*200, turning_points_trace_Jee[2], 'o', markerfacecolor=pal[2], markeredgecolor='none', markersize=marker_size*1.5)
# plt.plot(turning_points_det_re[0]*200, turning_points_det_Jee[0], 'o', markerfacecolor=pal[0], markeredgecolor='none', markersize=marker_size)
# plt.plot(turning_points_det_re[1]*200, turning_points_det_Jee[1], 'o', markerfacecolor=pal[1], markeredgecolor='none', markersize=marker_size)
# plt.plot(turning_points_det_re[2]*200, turning_points_det_Jee[2], 'o', markerfacecolor=pal[2], markeredgecolor='none', markersize=marker_size)

plt.xticks(np.arange(0, 10001, 5000), [0, 25, 50], fontsize=font_size_1, **hfont)
plt.yticks([0, 1, 2, 3], fontsize=font_size_1, **hfont)

plt.xlim([0, len(l_re)-1])
plt.ylim([0, 3])
plt.xlabel(r'$r_E$', fontsize=font_size_1, **hfont)
plt.ylabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
plt.legend([r'$u_d$: 0.2', r'$u_d$: 0.6', r'$u_d$: 1.0'], prop={"family": "Arial", 'size': font_size_1}, loc='best')
plt.savefig('paper_figures/png/Fig_S3_Supralinear_networks_EE_STD_ISN_boundary_turning_point_u_d.png')
plt.savefig('paper_figures/pdf/Fig_S3_Supralinear_networks_EE_STD_ISN_boundary_turning_point_u_d.pdf')
