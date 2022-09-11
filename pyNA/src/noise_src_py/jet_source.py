import pdb
import openmdao
import openmdao.api as om
import numpy as np


def jet_mixing_source(source, theta, inputs: openmdao.vectors.default_vector.DefaultVector) -> np.ndarray:
	"""
	Compute jet mixing noise mean-square acoustic pressure (msap).

	:param source: pyNA component computing noise sources
	:type source: Source
	:param inputs: unscaled, dimensional input variables read via inputs[key]
	:type inputs: openmdao.vectors.default_vector.DefaultVector

	:return: msap_jet_mixing
	:rtype: np.ndarray

	"""
	# Load options
	data = source.options['data']
	settings = source.options['settings']
	airframe = source.options['airframe']
	n_t = source.options['n_t']

	# Extract inputs
	V_j_star = inputs['V_j_star']
	rho_j_star = inputs['rho_j_star']
	A_j_star = inputs['A_j_star']
	Tt_j_star = inputs['Tt_j_star']
	M_0 = inputs['M_0']
	c_0 = inputs['c_0']
	r_s_star = 0.3048 / np.sqrt(settings['A_e'])
	jet_delta = 0.

	# Initialize solution
	msap_jet_mixing = np.zeros((n_t, settings['n_frequency_bands']))

	# Calculate jet mixing
	for i in np.arange(n_t):
		# Calculate density exponent (omega)
		# Source: Zorumski report 1982 part 2. Chapter 8.4 Table II
		log10Vja0 = np.log10(V_j_star[i])
		if -0.45 < log10Vja0 < 0.25:
			array_1 = np.array([-0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25])
			array_2 = np.array([-1, -0.9, -0.76, -0.58, -0.41, -0.22, 0, 0.22, 0.5, 0.77, 1.07, 1.39, 1.74, 1.95, 2])
			omega = np.interp(log10Vja0, array_1, array_2)
		elif log10Vja0 >= 0.25:
			omega = 2
		else:
			omega = np.NaN

		# Calculate power deviation factor (P)
		# Source: Zorumski report 1982 part 2. Chapter 8.4 Table III
		if -0.4 < log10Vja0 < 0.4:
			array_1 = np.array([-0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,0.4])
			array_2 = np.array([-0.13, -0.13, -0.13, -0.13, -0.13, -0.12, -0.1, -0.05, 0, 0.1, 0.21, 0.32, 0.41, 0.43, 0.41,0.31, 0.14])
			log10P = np.interp(log10Vja0, array_1, array_2)
			P_function = 10 ** log10P
		else:
			P_function = np.NaN

		# Calculate acoustic power (Pi_star)
		# Source: Zorumski report 1982 part 2. Chapter 8.4 Equation 3
		K = 6.67e-5
		Pi_star = K * rho_j_star[i] ** omega * V_j_star[i] ** 8 * P_function

		# Calculate directivity function (D)
		# Source: Zorumski report 1982 part 2. Chapter 8.4 Table IV
		if -0.4 < log10Vja0 < 0.4 and 0 <= theta[i] <= 180:
			log10D = data.jet_D_f(log10Vja0, theta[i])
			D_function = 10 ** log10D
		else:
			D_function = np.NaN
			raise ValueError('Outside domain.')

		# Calculate Strouhal frequency adjustment factor (xi)
		# Source: Zorumski report 1982 part 2. Chapter 8.4 Table V
		if V_j_star[i] <= 1.4:
			xi = 1
		else:
			if theta[i] <= 120:
				xi = 1.
			elif 120 < theta[i] <= 180:
				xi = data.jet_xi_f(V_j_star[i], theta[i])
				xi = min(1, xi)
			else:
				xi = np.NaN

		# Calculate Strouhal number (St)
		# Source: Zorumski report 1982 part 2. Chapter 8.4 Eq. 9
		D_j_star = np.sqrt(4 * A_j_star[i] / np.pi)  # Jet diamater [-] (rel. to sqrt(settings['A_e']))
		f_star = data.f * np.sqrt(settings['A_e']) / c_0[i]
		St = (f_star * D_j_star) / (xi * (V_j_star[i] - M_0[i]))
		log10St = np.log10(St)
		# Source: Zorumski report 1982 part 2. Chapter 8.4 Table VI
		if 1 <= Tt_j_star[i] <= 3.5:
			mlog10F = data.jet_F_f((theta[i]*np.ones(settings['n_frequency_bands']), Tt_j_star[i]*np.ones(settings['n_frequency_bands']), log10Vja0*np.ones(settings['n_frequency_bands']), log10St))

		# Add linear extrapolation for jet temperature
		# Computational fix for data unavailability
		elif Tt_j_star[i] > 3.5:
			point_a = (theta[i]*np.ones(settings['n_frequency_bands']), 3.5*np.ones(settings['n_frequency_bands']), log10Vja0*np.ones(settings['n_frequency_bands']), log10St)
			point_b = (theta[i]*np.ones(settings['n_frequency_bands']), 3.4*np.ones(settings['n_frequency_bands']), log10Vja0*np.ones(settings['n_frequency_bands']), log10St)
			mlog10F_a = data.jet_F_f(point_a)
			mlog10F_b = data.jet_F_f(point_b)

			# Extrapolation for the temperature data
			mlog10F = (mlog10F_a - mlog10F_b) / 0.1 * (Tt_j_star[i] - 3.5) + mlog10F_a
		else:
			point_a = (theta[i] * np.ones(settings['n_frequency_bands']), 1.1 * np.ones(settings['n_frequency_bands']), log10Vja0 * np.ones(settings['n_frequency_bands']), log10St)
			point_b = (theta[i] * np.ones(settings['n_frequency_bands']), 1.0 * np.ones(settings['n_frequency_bands']), log10Vja0 * np.ones(settings['n_frequency_bands']), log10St)
			mlog10F_a = data.jet_F_f(point_a)
			mlog10F_b = data.jet_F_f(point_b)

			# Extrapolation for the temperature data
			mlog10F = (mlog10F_a - mlog10F_b) / 0.1 * (Tt_j_star[i] - 1.0) + mlog10F_b
		F_function = 10 ** (-mlog10F / 10)

		# Calculate forward velocity index (m_theta)
		# Source: Zorumski report 1982 part 2. Chapter 8.4 Table VII
		array_1 = np.linspace(0, 180, 19)
		array_2 = np.array([3., 1.65, 1.1, 0.5, 0.2, 0.0, 0.0, 0.1, 0.4, 1, 1.9, 3, 4.7, 7, 8.5, 8.5, 8.5, 8.5, 8.5])
		m_theta = np.interp(theta[i], array_1, array_2)

		# Calculate mean-square acoustic pressure (msap)
		# Source: Zorumski report 1982 part 2. Chapter 8.4 Equation 8
		msap_j = Pi_star * A_j_star[i] / (4 * np.pi * r_s_star ** 2) * D_function * F_function / (1 - M_0[i] * np.cos(np.pi / 180. * (theta[i] - jet_delta))) * ((V_j_star[i] - M_0[i]) / V_j_star[i]) ** m_theta

		# Multiply with number of engines
		msap_j = msap_j * airframe.n_eng

		# Normalize msap by reference pressure
		msap_jet_mixing[i,:] = msap_j/settings['p_ref']**2
	
	return msap_jet_mixing

def jet_shock_source(source, theta, inputs: openmdao.vectors.default_vector.DefaultVector) -> np.ndarray:
	"""
	Compute jet mixing noise mean-square acoustic pressure (msap).

	:param source:
	:type source:
	:param inputs: unscaled, dimensional input variables read via inputs[key]
	:type inputs: openmdao.vectors.default_vector.DefaultVector

	:return: msap_jet_shock
	:rtype: [n_t, settings['n_frequency_bands']]
	"""
	# Load options
	settings = source.options['settings']
	data = source.options['data']
	airframe = source.options['airframe']
	n_t = source.options['n_t']

	# Extract inputs
	V_j_star = inputs['V_j_star']
	M_j = inputs['M_j']
	A_j_star = inputs['A_j_star']
	Tt_j_star = inputs['Tt_j_star']
	M_0 = inputs['M_0']
	c_0 = inputs['c_0']
	r_s_star = settings['r_0'] / np.sqrt(settings['A_e'])
	jet_delta = 0.

	# Initialize solution
	msap_jet_shock = np.zeros((n_t, settings['n_frequency_bands']))

	# Calculate jet shock
	for i in np.arange(n_t):
		# Calculate msap for all frequencies
		# If the jet is supersonic: shock cell noise
		if M_j[i] > 1:
			# Calculate beta function
			# Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 4
			beta = (M_j[i] ** 2 - 1) ** 0.5

			# Calculate eta (exponent of the pressure ratio parameter)
			# Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 5
			if beta > 1:
				if Tt_j_star[i] < 1.1:
					eta = 1.
				else:
					eta = 2.
			else:
				eta = 4.

			# Calculate f_star
			# Source: Zorumski report 1982 part 2. Chapter 8.5 page 8-5-1 (symbols)
			f_star = data.f * np.sqrt(settings['A_e']) / c_0[i]

			# Calculate sigma parameter
			# Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 3
			sigma = 7.80 * beta * (1 - M_0[i] * np.cos(np.pi / 180 * theta[i])) * np.sqrt(A_j_star[i]) * f_star
			log10sigma = np.log10(sigma)

			# Calculate C function
			# Source: Zorumski report 1982 part 2. Chapter 8.5 Table II
			array_1_c = np.array([-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2])
			array_2_c = np.array([0.703, 0.703, 0.71, 0.714, 0.719, 0.724, 0.729, 0.735, 0.74, 0.74, 0.74, 0.735, 0.714,0.681, 0.635, 0.579, 0.52, 0.46, 0.4, 0.345, 0.29, 0.235, 0.195, 0.15, 0.1, 0.06, 0.03, 0.015])

			# Calculate W function
			# Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 6-7
			b = 0.23077
			W = 0
			for k in np.arange(1, settings['n_shock']):
				sum_inner = 0
				for m in np.arange(settings['n_shock'] - k):
					# Calculate q_km
					q_km = 1.70 * k / V_j_star[i] * (1 - 0.06 * (m + (k + 1) / 2)) * (1 + 0.7 * V_j_star[i] * np.cos(np.pi / 180 * theta[i]))

					# Calculate inner sum (note: the factor b in the denominator below the sine should not be there: to get same graph as Figure 4)
					sum_inner = sum_inner + np.sin((b * sigma * q_km / 2)) / (sigma * q_km) * np.cos(sigma * q_km)

				# Compute the correlation coefficient spectrum C
				C = np.interp(log10sigma, array_1_c, array_2_c)
				# C = 10**log10C

				# Add outer loop to the shock cell interference function
				W = W + (4. / (settings['n_shock'] * b))* sum_inner * C ** (k ** 2)

			# Calculate the H function
			# Source: Zorumski report 1982 part 2. Chapter 8.5 Table III (+ linear extrapolation in logspace for log10sigma < 0; as given in SAEARP876)
			array_1_H = np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5])
			array_2_H = np.array([-5.73, -5.35, -4.97, -4.59, -4.21, -3.83, -3.45, -3.07, -2.69, -2.31, -1.94, -1.59, -1.33,-1.1, -0.94, -0.88, -0.91, -0.99, -1.09, -1.17, -1.3, -1.42, -1.55, -1.67, -1.81, -1.92, -2.06, -2.18, -2.3, -2.42, -2.54, -2.66, -2.78, -2.9])
			log10H = np.interp(log10sigma, array_1_H, array_2_H)

			# Source: Zorumski report 1982 part 2. Chapter 8.5.4
			if Tt_j_star[i] < 1.1:
				log10H = log10H - 0.2
			H = (10 ** log10H)

			# Calculate mean-square acoustic pressure (msap)
			# Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 1
			msap_j = 1.92e-3 * A_j_star[i] / (4 * np.pi * r_s_star ** 2) * (1 + W) / (1 - M_0[i] * np.cos(np.pi / 180. * (theta[i] - jet_delta))) ** 4 * beta ** eta * H
		else:
			msap_j = np.zeros(settings['n_frequency_bands']) * M_j[i] ** 0

		# Multiply with number of engines
		msap_j = msap_j * airframe.n_eng

		# Normalize msap by reference pressure
		msap_jet_shock[i,:] = msap_j/settings['p_ref']**2

	return msap_jet_shock


