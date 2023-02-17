import pdb
import openmdao
import openmdao.api as om
import numpy as np


def core_source(source, theta, inputs: openmdao.vectors.default_vector.DefaultVector) -> np.ndarray:
	"""
	Compute core noise mean-square acoustic pressure (msap).

	:param source: pyNA component computing noise sources
	:type source: Source
	:param inputs: unscaled, dimensional input variables read via inputs[key]
	:type inputs: openmdao.vectors.default_vector.DefaultVector

	:return: msap_core
	:rtype: np.ndarray [n_t, settings['n_frequency_bands']]
	"""
	# Load options
	settings = source.options['settings']
	data = source.options['data']
	airframe = source.options['airframe']
	n_t = source.options['n_t']

	# Extract inputs
	M_0 = inputs['M_0']
	if settings['core_turbine_attenuation_method'] == 'ge':
		mdoti_c_star = inputs['mdoti_c_star']
		Tti_c_star = inputs['Tti_c_star']
		Ttj_c_star = inputs['Ttj_c_star']
		Pti_c_star = inputs['Pti_c_star']
		DTt_des_c_star = inputs['DTt_des_c_star']
		
	elif settings['core_turbine_attenuation_method'] == 'pw':
		mdoti_c_star = inputs['mdoti_c_star']
		Tti_c_star = inputs['Tti_c_star']
		Ttj_c_star = inputs['Ttj_c_star']
		Pti_c_star = inputs['Pti_c_star']
		rho_te_c_star = inputs['rho_te_c_star']
		c_te_c_star = inputs['c_te_c_star']
		rho_ti_c_star = inputs['rho_ti_c_star']
		c_ti_c_star = inputs['c_ti_c_star']
	else:
		raise ValueError('Invalid method for turbine noise in core module. Specify: GE/PW.')
	
	r_s_star = settings['r_0'] / np.sqrt(settings['A_e'])
	A_c_star = 1.

	# Initialize solution
	msap_core = np.zeros((n_t, settings['n_frequency_bands']))

	# Compute core
	for i in np.arange(n_t):

		# Turbine transmission loss function
		# Source: Zorumski report 1982 part 2. Chapter 8.2 Equation 3
		if settings['core_turbine_attenuation_method'] == 'ge':
			g_TT = DTt_des_c_star[i] ** (-4)
		# Source: Hultgren, 2012: A comparison of combustor models Equation 6
		elif settings['core_turbine_attenuation_method'] == 'pw':
			zeta = (rho_te_c_star[i] * c_te_c_star[i]) / (rho_ti_c_star[i] * c_ti_c_star[i])
			g_TT = 0.8 * zeta / (1 + zeta) ** 2
		else:
			raise ValueError('Invalid method to account for turbine attenuation effects of combustor noise. Specify GE/PW.')

		# Calculate acoustic power (Pi_star)
		# Source Zorumski report 1982 part 2. Chapter 8.2 Equation 3
		Pi_star = 8.85e-7 * (mdoti_c_star[i] / A_c_star) * ((Ttj_c_star[i] - Tti_c_star[i]) / Tti_c_star[i]) ** 2 * Pti_c_star[i] ** 2 * g_TT

		# Calculate directivity function (D)
		# Take the D function as SAE ARP876E Table 18 and all other values which are not in the table from Zorumski report 1982 part 2. Chapter 8.2 Table II
		array_1 = np.linspace(0, 180, 19)
		array_2 = np.array([-0.85, -0.8, -0.75, -0.7, -0.65, -0.6, -0.53, -0.46, -0.39, -0.16, 0.08, 0.31, 0.5, 0.35, 0.12,-0.19,-0.51, -0.8, -0.9])
		D_function = np.interp(theta[i], array_1, array_2)
		D_function = 10 ** D_function

		# Calculate the spectral function (S)
		# Source Zorumski report 1982 part 2. Chapter 8.2 Equation 4
		f_p = 400 / (1 - M_0[i] * np.cos(theta[i] * np.pi / 180.))

		log10ffp = np.log10(data.f / f_p)

		# Take the S function as SAE ARP876E Table 17 and all other values which are not in the table from Zorumski report 1982 part 2. Chapter 8.2 Table III
		array_1 = np.linspace(-1.1, 1.6, 28)
		array_2 = np.array([-3.87, -3.47, -3.12, -2.72, -2.32, -1.99, -1.7, -1.41, -1.17, -0.97, -0.82, -0.72, -0.82, -0.97,
							-1.17, -1.41, -1.7, -1.99, -2.32, -2.72, -3.12, -3.47, -3.87, -4.32, -4.72, -5.22, -5.7, -6.2])
		S_function = np.interp(log10ffp, array_1, array_2)
		S_function = 10 ** S_function

		# Calculate mean-square acoustic pressure (msap)
		# Source Zorumski report 1982 part 2. Chapter 8.2 Equation 1
		msap_j = Pi_star * A_c_star / (4 * np.pi * r_s_star ** 2) * D_function * S_function / (1. - M_0[i] * np.cos(np.pi / 180. * theta[i])) ** 4

		# Multiply with number of engines
		msap_j = msap_j * airframe.n_eng

		# Normalize msap by reference pressure
		msap_core[i, :] = msap_j/settings['p_ref']**2

	return msap_core
