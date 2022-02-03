import matplotlib.pyplot as plt
import numpy as np

def load_icao_noise_limits(mtow: float, chapter: str, n_eng: int):

	limits = dict()
	limits['lateral'] = np.zeros(np.size(mtow))
	limits['flyover'] = np.zeros(np.size(mtow))
	limits['approach'] = np.zeros(np.size(mtow))

	# ICAO Chapter 3 limits
	if chapter == '3':
		limits['lateral'][mtow <= 35] = 94*np.ones(np.size(mtow))[mtow <= 35]
		limits['lateral'][(35<mtow)*(mtow<=400)]= (80.87 + 8.51*np.log10(mtow))[(35 < mtow)*(mtow<= 400)]
		limits['lateral'][400<mtow]=103*np.ones(np.size(mtow))[400 < mtow]

		limits['approach'][mtow<=35] = 98*np.ones(np.size(mtow))[mtow <= 35]
		limits['approach'][(35<mtow)*(mtow<=280)]= (86.03 + 7.75*np.log10(mtow))[(35 < mtow)*(mtow<=280)]
		limits['approach'][280<mtow]=105*np.ones(np.size(mtow))[280<mtow]

		if n_eng == 2:
		    limits['flyover'][mtow <= 48.1] = 89*np.ones(np.size(mtow))[mtow <= 48.1] 
		    limits['flyover'][(48.1 < mtow)*(mtow<= 385)] = (66.65 + 13.29*np.log10(mtow))[(48.1 < mtow)*(mtow<= 385)]
		    limits['flyover'][385 < mtow] = 101*np.ones(np.size(mtow))[385 < mtow] 
		elif n_eng == 3:
		    limits['flyover'][mtow <= 28.6] = 89*np.ones(np.size(mtow))[mtow <= 28.6] 
		    limits['flyover'][(28.6 < mtow)*(mtow<= 385)] = (69.65 + 13.29*np.log10(mtow))[(28.6 < mtow)*(mtow<= 385)]
		    limits['flyover'][385 < mtow] = 104*np.ones(np.size(mtow))[385 < mtow] 
		elif n_eng == 4:
		    limits['flyover'][mtow <= 20.2] = 89*np.ones(np.size(mtow))[mtow <= 20.2] 
		    limits['flyover'][(20.2 < mtow)*(mtow<= 385)] = (71.65 + 13.29*np.log10(mtow))[(20.2 < mtow)*(mtow<= 385)]
		    limits['flyover'][385 < mtow] = 106*np.ones(np.size(mtow))[385 < mtow] 
		else:
			raise ValueError("ICAO Chapter " + chapter + " noise limits not available for aircraft with " + str(n_eng) + " engines.")

	# ICAO Chapter 4 limits
	elif chapter == '14':
		limits['lateral'][mtow <= 2] = 88.6*np.ones(np.size(mtow))[mtow <= 2]
		limits['lateral'][(2 < mtow)*(mtow<= 8.618)]= (86.03754 + 8.512295*np.log10(mtow))[(2 < mtow)*(mtow<= 8.618)]
		limits['lateral'][(8.618 < mtow)*(mtow<= 35)]= 94*np.ones(np.size(mtow))[(8.618 < mtow)*(mtow<= 35)]
		limits['lateral'][(35 < mtow)*(mtow<= 400)]= (80.87 + 8.51*np.log10(mtow))[(35 < mtow)*(mtow<= 400)]
		limits['lateral'][400 < mtow] = 103*np.ones(np.size(mtow))[400 < mtow]

		limits['approach'][mtow <= 2] = 93.1*np.ones(np.size(mtow))[mtow <= 2]
		limits['approach'][(2 < mtow)*(mtow<= 8.618)]= (90.77481 + 7.72412*np.log10(mtow))[(2 < mtow)*(mtow<= 8.618)]
		limits['approach'][(8.618 < mtow)*(mtow<= 35)]= 98*np.ones(np.size(mtow))[(8.618 < mtow)*(mtow<= 35)]
		limits['approach'][(35<mtow)*(mtow<= 280)]= (86.03167 + 7.75117*np.log10(mtow))[(35 < mtow)*(mtow<= 280)]
		limits['approach'][280<mtow] = 105*np.ones(np.size(mtow))[280<mtow]

		if n_eng == 2:
		    limits['flyover'][mtow <= 2] = 80.6*np.ones(np.size(mtow))[mtow <= 2]
		    limits['flyover'][(2 < mtow)*(mtow<= 8.618)]= (76.57059 + 13.28771*np.log10(mtow))[(2 < mtow)*(mtow<= 8.618)]
		    limits['flyover'][(8.618 < mtow)*(mtow<=48.125)]= 89*np.ones(np.size(mtow))[(8.618 < mtow)*(mtow<=48.125)]
		    limits['flyover'][(48.125 < mtow)*(mtow<= 385)] = (66.65 + 13.29*np.log10(mtow))[(48.125 < mtow)*(mtow<= 385)]
		    limits['flyover'][385 < mtow] = 101*np.ones(np.size(mtow))[385 < mtow] 
		elif n_eng == 3:
		    limits['flyover'][mtow <= 2] = 80.6*np.ones(np.size(mtow))[mtow <= 2]
		    limits['flyover'][(2 < mtow)*(mtow<= 8.618)]= (76.57059 + 13.28771*np.log10(mtow))[(2 < mtow)*(mtow<= 8.618)]
		    limits['flyover'][(8.618 < mtow)*(mtow<= 28.615)]= 89*np.ones(np.size(mtow))[(8.618 < mtow)*(mtow<= 28.615)]
		    limits['flyover'][(28.615 < mtow)*(mtow<= 385)] = (69.65 + 13.29*np.log10(mtow))[(28.615 < mtow)*(mtow<= 385)]
		    limits['flyover'][385 < mtow] = 104*np.ones(np.size(mtow))[385 < mtow] 
		elif n_eng == 4:
		    limits['flyover'][mtow <= 2] = 80.6*np.ones(np.size(mtow))[mtow <= 2]
		    limits['flyover'][(2 < mtow)*(mtow<= 8.618)]= (76.57059 + 13.28771*np.log10(mtow))[(2 < mtow)*(mtow<= 8.618)]
		    limits['flyover'][(8.618 < mtow)*(mtow<= 20.234)]= 89*np.ones(np.size(mtow))[(8.618 < mtow)*(mtow<= 20.234)]
		    limits['flyover'][(20.234 < mtow)*(mtow<= 385)] = (71.65 + 13.29*np.log10(mtow))[(20.234 < mtow)*(mtow<= 385)]
		    limits['flyover'][385 < mtow] = 106*np.ones(np.size(mtow))[385 < mtow] 
		else:
			raise ValueError("ICAO Chapter " + chapter + " noise limits not available for aircraft with " + str(n_eng) + " engines.")

	# FAA NPRM noise limits
	elif chapter == 'NPRM':

		limits['lateral'][mtow <= 35] = 94*np.ones(np.size(mtow))[mtow <= 35]
		limits['lateral'][(35 < mtow)*(mtow<= 68.039)] = (80.87 + 8.51*np.log10(mtow))[(35 < mtow)*(mtow<= 68.039)]
		limits['lateral'][68.039 < mtow] = np.nan*np.ones(np.size(mtow))[68.039 < mtow]

		limits['approach'][mtow <= 35] = 98*np.ones(np.size(mtow))[mtow <= 35]
		limits['approach'][(35 < mtow)*(mtow<= 68.039)] = (86.03167 + 7.75117*np.log10(mtow))[(35 < mtow)*(mtow<= 68.039)]
		limits['approach'][68.039 < mtow] = np.nan*np.ones(np.size(mtow))[68.039 < mtow]

		if n_eng == 2:
		    limits['flyover'][mtow <= 48.125] = 89*np.ones(np.size(mtow))[mtow <= 48.125]
		    limits['flyover'][(48.125 < mtow)*(mtow<= 68.039)] = (66.65 + 13.29*np.log10(mtow))[(48.125 < mtow)*(mtow<=68.039)]
		    limits['flyover'][68.039 < mtow] = np.nan*np.ones(np.size(mtow))[68.039 < mtow]
		elif n_eng == 3:
		    limits['flyover'][mtow <= 28.615] = 89*np.ones(np.size(mtow))[mtow <= 28.615]
		    limits['flyover'][(28.615 < mtow)*(mtow<= 68.039)] = (69.65 + 13.29*np.log10(mtow))[(28.615 < mtow)*(mtow<= 68.039)]
		    limits['flyover'][68.039 < mtow] = np.nan*np.ones(np.size(mtow))[68.039 < mtow]
		else:
			raise ValueError("ICAO Chapter " + chapter + " noise limits not available for aircraft with " + str(n_eng) + " engines.")

	else:
		raise ValueError("ICAO Chapter " + chapter + "noise limits are not available. Specify '3', '14', or 'NPRM'.")

	return limits

if __name__ == "__main__":

	# Max take-off weight
	mtow = np.logspace(0, np.log10(1000), 1000)

	limits = dict()
	limits['3'] = dict()
	limits['3']['2'] = load_icao_noise_limits(mtow, '3', 2)
	limits['3']['3'] = load_icao_noise_limits(mtow, '3', 3)
	limits['3']['4'] = load_icao_noise_limits(mtow, '3', 4)

	limits['14'] = dict()
	limits['14']['2'] = load_icao_noise_limits(mtow, '14', 2)
	limits['14']['3'] = load_icao_noise_limits(mtow, '14', 3)
	limits['14']['4'] = load_icao_noise_limits(mtow, '14', 4)

	limits['NPRM'] = dict()
	limits['NPRM']['2'] = load_icao_noise_limits(mtow, 'NPRM', 2)
	limits['NPRM']['3'] = load_icao_noise_limits(mtow, 'NPRM', 3)

	# Plot ICAO noise limits
	fig, ax = plt.subplots(1,1,figsize=(10, 4), dpi=100)
	plt.style.use('plot.mplstyle')

	plt.semilogx(mtow, limits['3']['2']['lateral'] + limits['3']['2']['flyover'] + limits['3']['2']['approach'], '-' ,color='tab:blue', label='Chapter 3')
	plt.semilogx(mtow, limits['3']['3']['lateral'] + limits['3']['3']['flyover'] + limits['3']['3']['approach'], '--',color='tab:blue')
	plt.semilogx(mtow, limits['3']['4']['lateral'] + limits['3']['4']['flyover'] + limits['3']['4']['approach'], '-.',color='tab:blue')

	plt.semilogx(mtow, limits['3']['2']['lateral'] + limits['3']['2']['flyover'] + limits['3']['2']['approach'] - 10, '-' , color='tab:orange', label='Chapter 4')
	plt.semilogx(mtow, limits['3']['3']['lateral'] + limits['3']['3']['flyover'] + limits['3']['3']['approach'] - 10, '--', color='tab:orange')
	plt.semilogx(mtow, limits['3']['4']['lateral'] + limits['3']['4']['flyover'] + limits['3']['4']['approach'] - 10, '-.', color='tab:orange')

	plt.semilogx(mtow, limits['14']['2']['lateral'] + limits['14']['2']['flyover'] + limits['14']['2']['approach'] - 17, '-' , color='tab:green', label='Chapter 14')
	plt.semilogx(mtow, limits['14']['3']['lateral'] + limits['14']['3']['flyover'] + limits['14']['3']['approach'] - 17, '--', color='tab:green')
	plt.semilogx(mtow, limits['14']['4']['lateral'] + limits['14']['4']['flyover'] + limits['14']['4']['approach'] - 17, '-.', color='tab:green')

	plt.plot([210,300],[259.5,259.5], '-', color='k')
	plt.plot([210,300],[253.5,253.5], '--', color='k')
	plt.plot([210,300],[247.5,247.5], '-.', color='k')

	plt.annotate(xy=(350,258), s='2 engines', fontsize=14)
	plt.annotate(xy=(350,252), s='3 engines', fontsize=14)
	plt.annotate(xy=(350,246), s='4 engines', fontsize=14)

	plt.xlabel('MTOW [$10^3$ kg]')
	plt.ylabel('$EPNL_{cumulative}$  [EPNdB]')
	plt.legend(fontsize=14, loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=3, borderaxespad=0, frameon=False)
	plt.xlim([1, 1000])
	plt.ylim([240, 320])
	plt.show()
	