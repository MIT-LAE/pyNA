import jax.numpy as jnp
import pdb


def core_emmerling(core_mdot, core_Tt_i, core_Tt_j, core_Pt_i, turb_DTt_des, turb_rho_e, turb_c_e, turb_rho_i, turb_c_i, M_0, rho_0, c_0, T_0, P_0, theta, f, settings, aircraft, tables):

    """
	Compute core noise mean-square acoustic pressure (msap).

	:param source: pyNA component computing noise sources
	:type source: Source
	:param inputs: unscaled, dimensional input variables read via inputs[key]
	:type inputs: openmdao.vectors.default_vector.DefaultVector

	:return: msap
	:rtype: jnp.ndarray [settings['n_frequency_bands'],]
	"""
	
    # Normalize engine inputs
    core_mdot_star = core_mdot/(rho_0*c_0*settings['A_e'])
    core_Tt_i_star = core_Tt_i/T_0
    core_Tt_j_star = core_Tt_j/T_0
    core_Pt_i_star = core_Pt_i/P_0
    turb_DTt_des_star = turb_DTt_des/T_0
    turb_rho_i_star = turb_rho_i/rho_0
    turb_c_i_star = turb_c_i/c_0
    turb_rho_e_star = turb_rho_e/rho_0
    turb_c_e_star = turb_c_e/c_0

	# Extract inputs
    r_s_star = settings['r_0'] / jnp.sqrt(settings['A_e'])
    A_c_star = 1.

    # Turbine transmission loss function
    # Source: Zorumski report 1982 part 2. Chapter 8.2 Equation 3
    if settings['core_turbine_attenuation_method'] == 'ge':
        g_TT = turb_DTt_des_star ** (-4)
    # Source: Hultgren, 2012: A comparison of combustor models Equation 6
    elif settings['core_turbine_attenuation_method'] == 'pw':
        zeta = (turb_rho_e_star * turb_c_e_star) / (turb_rho_i_star * turb_c_i_star)
        g_TT = 0.8 * zeta / (1 + zeta) ** 2
    else:
        raise ValueError('Invalid method to account for turbine attenuation effects of combustor noise. Specify GE/PW.')

    # Calculate acoustic power (Pi_star)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 3
    Pi_star = 8.85e-7 * (core_mdot_star / A_c_star) * ((core_Tt_j_star - core_Tt_i_star) / core_Tt_i_star) ** 2 * core_Pt_i_star ** 2 * g_TT

    # Calculate directivity function (D)
    # Take the D function as SAE ARP876E Table 18 and all other values which are not in the table from Zorumski report 1982 part 2. Chapter 8.2 Table II
    D_function = jnp.interp(theta, tables.core.D_theta, tables.core.D_data)
    D_function = 10 ** D_function

    # Calculate the spectral function (S)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 4
    f_p = 400 / (1 - M_0 * jnp.cos(theta * jnp.pi / 180.))
    log10ffp = jnp.log10(f / f_p)

    # Take the S function as SAE ARP876E Table 17 and all other values which are not in the table from Zorumski report 1982 part 2. Chapter 8.2 Table III
    S_function = jnp.interp(log10ffp, tables.core.S_log10ffp, tables.core.S_data)
    S_function = 10 ** S_function

    # Calculate mean-square acoustic pressure (msap)
    # Multiply with number of engines and normalize msap by reference pressure
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 1
    msap = Pi_star * A_c_star / (4 * jnp.pi * r_s_star ** 2) * D_function * S_function / (1. - M_0 * jnp.cos(jnp.pi / 180. * theta)) ** 4 * (aircraft.n_eng/settings['p_ref']**2)

    return msap