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
l_u_d = [1.0]
tau_x = 0.20
tau_e = 0.020

x_max = 2
l_re = np.arange(0, x_max+0.0001, 0.001)

alpha_e, alpha_i = 0.5, 0.5
boundary_value_ISN = np.zeros((len(l_u_d), len(l_re))) * np.nan
boundary_value_pd = np.zeros((len(l_u_d), len(l_re))) * np.nan
trace_boundary_value = np.zeros((len(l_u_d), len(l_re))) * np.nan
det_boundary_value = np.zeros((len(l_u_d), len(l_re))) * np.nan

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

for u_d_idx in range(len(l_u_d)):
    for re_idx in range(len(l_re)):
        if det_boundary_value[u_d_idx, re_idx] < trace_boundary_value[u_d_idx, re_idx]:
            boundary_value_ISN[u_d_idx, re_idx] = det_boundary_value[u_d_idx, re_idx]
        else:
            boundary_value_ISN[u_d_idx, re_idx] = trace_boundary_value[u_d_idx, re_idx]

for u_d_idx in range(len(l_u_d)):
    u_d = l_u_d[u_d_idx]
    for re_idx in range(len(l_re)):
        re = l_re[re_idx]

        string = """ {solve_for} /. Solve[{{ - (Jee * {tau_x} * {u_d} * {re})/((1 + {tau_x} * {u_d} * {re})^2) + Jee /(1 + {tau_x} * {u_d} * {re}) - 1/{alpha_e} * ({re}^(1/{alpha_e}-1))==0}}, {solve_for}, Reals]""".format(
            tau_e=tau_e, tau_x=tau_x, u_d=u_d, re=re, alpha_e=alpha_e, alpha_i=alpha_i, solve_for='{Jee}')

        P = session.evaluate(string)
        if len(P) == 1:
            solution_string = str(P[0])
            solution_Jee = solution_string.split(",")[0]
            if solution_Jee != 'Global`Jee':
                Jee_solution = float(solution_Jee[1:])
                boundary_value_pd[u_d_idx, re_idx] = Jee_solution
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

plt.plot(boundary_value_ISN[0, :], linewidth=plot_line_width)

plt.plot(boundary_value_pd[0, :], color=pal[0], linestyle='dashed', linewidth=plot_line_width)

plt.xticks(np.arange(0, 2001, 1000), [0, 1, 2], fontsize=font_size_1, **hfont)
plt.yticks([0, 1, 2, 3], fontsize=font_size_1, **hfont)

plt.xlim([0, len(l_re)-1])
plt.ylim([0, 3])
plt.xlabel(r'$r_E$', fontsize=font_size_1, **hfont)
plt.ylabel(r'$J_{EE}$', fontsize=font_size_1, **hfont)
plt.legend([r'$u_d$: 1.0'], prop={"family": "Arial", 'size': font_size_1}, loc='best')
plt.savefig('paper_figures/png/Fig_S6_ISN_index_paradoxical_effect_sublinear_networks_EE_STD_boundary.png')
plt.savefig('paper_figures/pdf/Fig_S6_ISN_index_paradoxical_effect_sublinear_networks_EE_STD_boundary.pdf')