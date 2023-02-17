py.settings.case_name = 'nasa_stca_standard'
py.noise.data.load_trajectory_verification_data(settings=py.settings)

# Plot 1
t_observer_stca = py.noise.data.verification_trajectory['flyover']['t observer [s]'].values
pnlt_stca = py.noise.data.verification_trajectory['flyover']['PNLT'].values
TS_stca = np.interp(t_observer_stca, nasa_std['t_source [s]'], nasa_std['TS [-]'])
t_epnl = t_observer_stca[np.where(pnlt_stca > np.max(pnlt_stca)-10)[0]]

fig, ax = plt.subplots(1, 1, figsize=(15,3), dpi=200)
plt.style.use('../utils/plot.mplstyle')
ax.plot(t_observer_stca, TS_stca)
ax.set_xlabel('Time after brake release [s]')
ax.set_xlim([15, 150])
ax.set_ylim([-0.05, 1.05])
ax.set_ylabel('TS [-]')
ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.09), ncol=1, borderaxespad=0, frameon=False)

ax.plot([45, 45], [-5, 1.05], 'k', linewidth=1.5)
ax.annotate(xy=(50, 0.1), s='$t>t_{control}$', fontsize=16)
ax.annotate('', xy=(44.8, 0.14), xycoords='data', xytext=(50, 0.14), textcoords='data', arrowprops=dict(arrowstyle= '<-', color='k', lw=1.5, ls='-', mutation_scale=15))
ax.fill_between([t_epnl[-1], 150], [-5, -5], [100, 100], hatch='/', alpha=0)
ax.annotate(xy=(129, 0.1), s='Region of \nundetermined control', fontsize=16, ha='center', backgroundcolor='w')

fig, ax = plt.subplots(1, 1, figsize=(15,3), dpi=200)
ax.plot(t_observer_stca, pnlt_stca)
ax.set_xlabel('Time after brake release [s]')
ax.set_xlim([15, 150])
ax.set_ylim([-5, 105])
ax.set_ylabel('PNLT [TPNdB]')

ax.plot([t_epnl[0], t_epnl[0]], [-5, 105], 'k', linewidth=1.5)
ax.plot([t_epnl[-1], t_epnl[-1]], [-5, 105], 'k', linewidth=1.5)
ax.annotate('', xy=(t_epnl[0]-0.5, 40), xycoords='data', xytext=(t_epnl[-1]+0.5, 40), textcoords='data', arrowprops=dict(arrowstyle= '<->', color='k', lw=1.5, ls='-', mutation_scale=15))
ax.annotate(xy=((t_epnl[0]+t_epnl[-1])/2, 8), s='[$t_1$, $t_2$] s.t. PNLT > \nPNLT$_{max}$ - 10 TPNdB', fontsize=16, ha='center', backgroundcolor='w')
ax.fill_between([t_epnl[-1], 150], [-5, -5], [105, 105], hatch='/', alpha=0)

ax.annotate(xy=(129, 8), s='Region of \nundetermined control', fontsize=16, ha='center', backgroundcolor='w')

plt.subplots_adjust(hspace=0.25)


# Plot 2
t_observer_stca = py.noise.data.verification_trajectory['flyover']['t observer [s]'].values
pnlt_stca = py.noise.data.verification_trajectory['flyover']['PNLT'].values
TS_stca = np.interp(t_observer_stca, nasa_std['t_source [s]'], nasa_std['TS [-]'])
idx_epnl = np.where(pnlt_stca > np.max(pnlt_stca)-10)[0]
t_epnl = t_observer_stca[idx_epnl]

fig, ax = plt.subplots(1, 1, figsize=(15,3), dpi=200)
ax.plot(t_observer_stca, 10**(pnlt_stca/10))
ax.set_xlabel('Time after brake release [s]')
# ax.set_xlim([15, 150])
# ax.set_ylim([-5, 105])
ax.set_ylabel('$10^{0.1PNLT}$ [-]')

ax.fill_between(t_observer_stca, 10**(0.1*pnlt_stca), color='tab:orange', alpha=0.2)
ax.fill_between(t_epnl, 10**(0.1*pnlt_stca[idx_epnl]), hatch='//', alpha=0)


ax.fill_between([96, 102], [1.35e8, 1.35e8], [2.15e8, 2.15e8], color='tab:orange', alpha=0.2)
ax.fill_between([88, 92], [3.5e8, 3.5e8], [4.3e8, 4.3e8], alpha=0., hatch='//')
ax.annotate(xy=(103.5, 1.5e8), s='IPNLT', fontsize=16)
ax.annotate(xy=(93.5, 3.65e8), s='EPNL', fontsize=16)

ax.annotate('', xy=(87, 3.9e8), xycoords='data', xytext=(78.7, 3.9e8), textcoords='data', arrowprops=dict(arrowstyle= '->', color='k', lw=1., ls='-', mutation_scale=15))
ax.annotate('', xy=(89, 0.1e8), xycoords='data', xytext=(95, 1.65e8), textcoords='data', arrowprops=dict(arrowstyle= '<-', color='k', lw=1., ls='-', mutation_scale=15))

# ax.plot([t_epnl[0], t_epnl[0]], [-5, 105], 'k', linewidth=1.5)
# ax.plot([t_epnl[-1], t_epnl[-1]], [-5, 105], 'k', linewidth=1.5)
# ax.annotate('', xy=(t_epnl[0]-0.5, 40), xycoords='data', xytext=(t_epnl[-1]+0.5, 40), textcoords='data', arrowprops=dict(arrowstyle= '<->', color='k', lw=1.5, ls='-', mutation_scale=15))
# ax.annotate(xy=((t_epnl[0]+t_epnl[-1])/2, 8), s='[$t_1$, $t_2$] s.t. PNLT > \nPNLT$_{max}$ - 10 TPNdB', fontsize=16, ha='center', backgroundcolor='w')


# ax.annotate(xy=(129, 8), s='Region of \nundetermined control', fontsize=16, ha='center', backgroundcolor='w')

plt.subplots_adjust(hspace=0.25)