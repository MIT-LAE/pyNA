import jax.numpy as jnp


def jet_mixing_sae876(jet_V, jet_Tt, jet_A, jet_rho, M_0, c_0, T_0, rho_0, theta, f, settings, aircraft, tables):

    """
    Compute jet mixing noise mean-square acoustic pressure (msap).
    
    """

    # Extract inputs
    jet_V_star = jet_V/c_0
    jet_rho_star = jet_rho/rho_0
    jet_A_star = jet_A/settings['A_e']
    jet_Tt_star = jet_Tt/T_0

    r_s_star = 0.3048 / jnp.sqrt(settings['A_e'])
    jet_delta = 0.

    # Calculate density exponent (omega)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table II
    log10Vjc0 = jnp.log10(jet_V_star)
    if -0.45 < log10Vjc0 < 2.0:
        omega = jnp.interp(log10Vjc0, tables.source.jet.omega_log10Vjc0, tables.source.jet.omega_data)
    else:
        raise ValueError('log10Vjc0 is out of bounds: [-0.45, 2.0]')

    # Calculate power deviation factor (P)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table III
    if -0.4 < log10Vjc0 < 0.4:
        log10P = jnp.interp(log10Vjc0, tables.source.jet.P_log10Vjc0, tables.source.jet.P_data)
        P_function = 10 ** log10P
    else:
        raise ValueError('log10Vjc0 is out of bounds: [-0.40, 0.40]')

    # Calculate acoustic power (Pi_star)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Equation 3
    K = 6.67e-5
    Pi_star = K * jet_rho_star ** omega * jet_V_star ** 8 * P_function

    # Calculate directivity function (D)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table IV
    if -0.4 < log10Vjc0 < 0.4 and 0. <= theta <= 180.:
        log10D = tables.source.jet.D_interp(log10Vjc0, theta)
        D_function = 10 ** log10D
    else:
        raise ValueError('log10Vjc0 and theta are out of bounds: [-0.4, 0.4] and [0, 180]deg')

    # Calculate Strouhal frequency adjustment factor (xi)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table V
    if 0.4 < jet_V_star < 2.5  and 0 <= theta <= 180.:
        xi = tables.jet.xi_interp(jet_V_star, theta)
    else:
        raise ValueError('jet_V_star and theta are out of bounds: [0.4, 2.5] and [0, 180]deg')            

    # Calculate Strouhal number (St)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Eq. 9
    D_j_star = jnp.sqrt(4 * jet_A_star / jnp.pi)  # Jet diamater [-] (rel. to sqrt(settings['A_e']))
    f_star = f * jnp.sqrt(settings['A_e']) / c_0
    St = (f_star * D_j_star) / (xi * (jet_V_star - M_0))
    log10St = jnp.log10(St)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table VI
    if 1. <= jet_Tt_star <= 3.5:
        mlog10F = tables.jet.F_interp((theta*jnp.ones(settings['n_frequency_bands']), jet_Tt_star*jnp.ones(settings['n_frequency_bands']), log10Vjc0*jnp.ones(settings['n_frequency_bands']), log10St))

    # Add linear extrapolation for jet temperature
    # Computational fix for data unavailability
    elif jet_Tt_star > 3.5:
        point_a = (theta*jnp.ones(settings['n_frequency_bands']), 3.5*jnp.ones(settings['n_frequency_bands']), log10Vjc0*jnp.ones(settings['n_frequency_bands']), log10St)
        point_b = (theta*jnp.ones(settings['n_frequency_bands']), 3.4*jnp.ones(settings['n_frequency_bands']), log10Vjc0*jnp.ones(settings['n_frequency_bands']), log10St)
        mlog10F_a = tables.jet.F_interp(point_a)
        mlog10F_b = tables.jet.F_interp(point_b)
        mlog10F = (mlog10F_a - mlog10F_b) / 0.1 * (jet_Tt_star - 3.5) + mlog10F_a

    else:
        point_a = (theta * jnp.ones(settings['n_frequency_bands']), 1.1 * jnp.ones(settings['n_frequency_bands']), log10Vjc0 * jnp.ones(settings['n_frequency_bands']), log10St)
        point_b = (theta * jnp.ones(settings['n_frequency_bands']), 1.0 * jnp.ones(settings['n_frequency_bands']), log10Vjc0 * jnp.ones(settings['n_frequency_bands']), log10St)
        mlog10F_a = tables.jet.F_interp(point_a)
        mlog10F_b = tables.jet.F_interp(point_b)
        mlog10F = (mlog10F_a - mlog10F_b) / 0.1 * (jet_Tt_star - 1.0) + mlog10F_b

    F_function = 10 ** (-mlog10F / 10)

    # Calculate forward velocity index (m_theta)
    m_theta = jnp.interp(theta, tables.source.core.m_theta_theta, tables.source.core.m_theta_data)

    # Calculate mean-square acoustic pressure (msap)
    # Multiply with number of engines and normalize msap by reference pressure
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Equation 8
    msap = Pi_star * jet_A_star / (4 * jnp.pi * r_s_star ** 2) * D_function * F_function / (1 - M_0 * jnp.cos(jnp.pi / 180. * (theta - jet_delta))) * ((jet_V_star - M_0) / jet_V_star) ** m_theta * (aircraft.n_eng / settings['p_ref']**2)

    return msap