import jax.numpy as jnp


def trailing_edge_wing(settings, aircraft, M_0, c_0, rho_0, mu_0, theta, phi, f):
    """
    Compute wing trailing edge mean-square acoustic pressure (msap).

    :param settings: pyna settings
    :type settings
    :param aircraft: aircraft parameters
    :type aircraft
    :param M_0: ambient Mach number [-]
    :type M_0
    :param c_0: ambient speed of sound [m/s]
    :type c_0
    :param rho_0: ambient density [kg/m3]
    :type rho_0
    :param mu_0: ambient dynamic viscosity [kg/m/s]
    :type mu_0
    :param theta: polar directivity angle [deg]
    :type theta
    :param phi: azimuthal directivity angle [deg]
    :type phi
    :param f: 1/3rd octave frequency [Hz]
    :type f

    :return: msap_w
    :rtype

    """
    ### ---------------- Wing trailing-edge noise ----------------
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 5
    delta_w_star = 0.37 * (aircraft.af_S_w / aircraft.af_b_w ** 2) * (rho_0 * M_0 * c_0 * aircraft.af_S_w / (mu_0 * aircraft.af_b_w)) ** (-0.2)

    # Determine configuration constant and the sound power
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 7
    if aircraft.af_clean_w:
        K_w = 7.075e-6
    else:
        K_w = 4.464e-5
    Pi_star_w = K_w * M_0 ** 5 * delta_w_star

    # Determine directivity function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
    D_w = 4. * jnp.cos(phi * jnp.pi / 180.) ** 2 * jnp.cos(theta / 2 * jnp.pi / 180.) ** 2

    # Determine spectral distribution function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-11-12
    S_w = f * delta_w_star * aircraft.af_b_w / (M_0 * c_0) * (1 - M_0 * jnp.cos(theta * jnp.pi / 180.))
    if aircraft.af_delta_wing == 1:
        F_w = 0.613 * (10 * S_w) ** 4 * ((10 * S_w) ** 1.35 + 0.5) ** (-4)
    elif aircraft.af_delta_wing == 0:
        F_w = 0.485 * (10 * S_w) ** 4 * ((10 * S_w) ** 1.5 + 0.5) ** (-4)
    else:
        raise ValueError('Invalid delta-wing flag configuration specified. Specify: 0/1.')

    # Determine msap
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_s_star_af = settings['r_0']/aircraft.af_b_w
    msap_w = 1 / (4 * jnp.pi * r_s_star_af ** 2) / (1 - M_0 * jnp.cos(theta * jnp.pi / 180.)) ** 4 * (Pi_star_w * D_w * F_w)

    return msap_w

def trailing_edge_horizontal_tail(settings, aircraft, M_0, c_0, rho_0, mu_0, theta, phi, f):
    """
    Compute horizontal tail trailing edge mean-square acoustic pressure (msap).

    :param settings: pyna settings
    :type settings
    :param aircraft: aircraft parameters
    :type aircraft
    :param M_0: ambient Mach number [-]
    :type M_0
    :param c_0: ambient speed of sound [m/s]
    :type c_0
    :param rho_0: ambient density [kg/m3]
    :type rho_0
    :param mu_0: ambient dynamic viscosity [kg/m/s]
    :type mu_0
    :param theta: polar directivity angle [deg]
    :type theta
    :param phi: azimuthal directivity angle [deg]
    :type phi
    :param f: 1/3rd octave frequency [Hz]
    :type f

    :return: msap_h
    :rtype
    """

    # ---------------- Horizontal tail trailing-edge noise ----------------
    # Trailing edge noise of the horizontal tail
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 5
    delta_h_star = 0.37 * (aircraft.af_S_h / aircraft.af_b_h ** 2) * (rho_0 * M_0 * c_0 * aircraft.af_S_h / (mu_0 * aircraft.af_b_h)) ** (-0.2)

    # Determine configuration constant and the sound power
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 7
    if aircraft.af_clean_h:
        K_h = 7.075e-6
    else:
        K_h = 4.464e-5
    Pi_star_h = K_h * M_0 ** 5 * delta_h_star * (aircraft.af_b_h / aircraft.af_b_w) ** 2

    # Determine the directivity function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
    D_h = 4 * jnp.cos(phi * jnp.pi / 180.) ** 2 * jnp.cos(theta / 2 * jnp.pi / 180.) ** 2

    # Determine spectral distribution function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-11-12
    S_h = f * delta_h_star * aircraft.af_b_h / (M_0 * c_0) * (1 - M_0 * jnp.cos(theta * jnp.pi / 180.))
    F_h = 0.485 * (10 * S_h) ** 4 * ((10 * S_h) ** 1.5 + 0.5) ** (-4)

    # Determine msap
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_s_star_af = settings['r_0'] / aircraft.af_b_w
    msap_h = 1. / (4. * jnp.pi * r_s_star_af ** 2) / (1 - M_0 * jnp.cos(theta * jnp.pi / 180.)) ** 4 * (Pi_star_h * D_h * F_h)

    return msap_h

def trailing_edge_vertical_tail(settings, aircraft, M_0, c_0, rho_0, mu_0, theta, phi, f):
    """
    Compute vertical tail trailing edge mean-square acoustic pressure (msap).

    :param settings: pyna settings
    :type settings
    :param aircraft: aircraft parameters
    :type aircraft
    :param M_0: ambient Mach number [-]
    :type M_0
    :param c_0: ambient speed of sound [m/s]
    :type c_0
    :param rho_0: ambient density [kg/m3]
    :type rho_0
    :param mu_0: ambient dynamic viscosity [kg/m/s]
    :type mu_0
    :param theta: polar directivity angle [deg]
    :type theta
    :param phi: azimuthal directivity angle [deg]
    :type phi
    :param f: 1/3rd octave frequency [Hz]
    :type f

    :return: msap_h
    :rtype
    """

    ### ---------------- Vertical tail trailing-edge noise ----------------
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 5
    delta_v_star = 0.37 * (aircraft.af_S_v / aircraft.af_b_v ** 2) * (rho_0 * M_0 * c_0 * aircraft.af_S_v / (mu_0 * aircraft.af_b_v)) ** (-0.2)

    # Trailing edge noise of the vertical tail
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 7
    if aircraft.af_clean_v:
        K_v = 7.075e-6
    else:
        K_v = 4.464e-5
    Pi_star_v = K_v * M_0 ** 5 * delta_v_star * (aircraft.af_b_v / aircraft.af_b_w) ** 2

    # Determine directivity function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
    D_v = 4 * jnp.sin(phi * jnp.pi / 180.) ** 2 * jnp.cos(theta / 2 * jnp.pi / 180.) ** 2

    # Determine spectral distribution function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-11-12
    S_v = f * delta_v_star * aircraft.af_b_v / (M_0 * c_0) * (1 - M_0 * jnp.cos(theta * jnp.pi / 180.))
    if aircraft.af_delta_wing:
        F_v = 0.613 * (10 * S_v) ** 4 * ((10 * S_v) ** 1.35 + 0.5) ** (-4)
    else:
        F_v = 0.485 * (10 * S_v) ** 4 * ((10 * S_v) ** 1.35 + 0.5) ** (-4)

    # Determine msap
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_s_star_af = settings['r_0'] / aircraft.af_b_w
    msap_v = 1. / (4 * jnp.pi * r_s_star_af ** 2) / (1 - M_0 * jnp.cos(theta * jnp.pi / 180.)) ** 4 * (Pi_star_v * D_v * F_v)

    return msap_v

def leading_edge_slat(settings, aircraft, M_0, c_0, rho_0, mu_0, theta, phi, f):
    """
    Compute leading-edge slat mean-square acoustic pressure (msap).

    :param settings: pyna settings
    :type settings
    :param aircraft: aircraft parameters
    :type aircraft
    :param M_0: ambient Mach number [-]
    :type M_0
    :param c_0: ambient speed of sound [m/s]
    :type c_0
    :param rho_0: ambient density [kg/m3]
    :type rho_0
    :param mu_0: ambient dynamic viscosity [kg/m/s]
    :type mu_0
    :param theta: polar directivity angle [deg]
    :type theta
    :param phi: azimuthal directivity angle [deg]
    :type phi
    :param f: 1/3rd octave frequency [Hz]
    :type f

    :return: msap_les
    :rtype
    """

    ### ---------------- Slat noise ----------------
    delta_w_star = 0.37 * (aircraft.af_S_w / aircraft.af_b_w ** 2) * (rho_0 * M_0 * c_0 * aircraft.af_S_w / (mu_0 * aircraft.af_b_w)) ** (-0.2)

    # Noise power
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 4
    Pi_star_les1 = 4.464e-5 * M_0 ** 5 * delta_w_star  # Slat noise
    Pi_star_les2 = 4.464e-5 * M_0 ** 5 * delta_w_star  # Added trailing edge noise
    # Determine the directivity function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
    D_les = 4 * jnp.cos(phi * jnp.pi / 180.) ** 2 * jnp.cos(theta / 2 * jnp.pi / 180.) ** 2
    # Determine spectral distribution function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-12-13
    S_les = f * delta_w_star * aircraft.af_b_w / (M_0 * c_0) * (1 - M_0 * jnp.cos(theta * jnp.pi / 180.))
    F_les1 = 0.613 * (10 * S_les) ** 4 * ((10. * S_les) ** 1.5 + 0.5) ** (-4)
    F_les2 = 0.613 * (2.19 * S_les) ** 4 * ((2.19 * S_les) ** 1.5 + 0.5) ** (-4)
    # Calculate msap
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_s_star_af = settings['r_0'] / aircraft.af_b_w
    msap_les = 1 / (4 * jnp.pi * r_s_star_af ** 2) / (1 - M_0 * jnp.cos(theta * jnp.pi / 180.)) ** 4 * (Pi_star_les1 * D_les * F_les1 + Pi_star_les2 * D_les * F_les2)

    return msap_les

def trailing_edge_flap(settings, aircraft, M_0, c_0, theta, phi, theta_flaps, f):
    """
    Compute trailing-edge flap mean-square acoustic pressure (msap).

    :param settings: pyna settings
    :type settings
    :param aircraft: aircraft parameters
    :type aircraft
    :param M_0: ambient Mach number [-]
    :type M_0
    :param c_0: ambient speed of sound [m/s]
    :type c_0
    :param theta: polar directivity angle [deg]
    :type theta
    :param phi: azimuthal directivity angle [deg]
    :type phi
    :param theta_flaps: flap deflection angle [deg]
    :type theta_flaps
    :param f: 1/3rd octave frequency [Hz]
    :type f

    :return: msap_tef
    :rtype
    """
    ### ---------------- Flap noise ----------------
    # Calculate noise power
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 14-15
    if aircraft.af_s < 3:
        Pi_star_tef = 2.787e-4 * M_0 ** 6 * aircraft.af_S_f / aircraft.af_b_w ** 2 * jnp.sin(theta_flaps * jnp.pi / 180.) ** 2
    elif aircraft.af_s == 3:
        Pi_star_tef = 3.509e-4 * M_0 ** 6 * aircraft.af_S_f / aircraft.af_b_w ** 2 * jnp.sin(theta_flaps * jnp.pi / 180.) ** 2
    else:
        raise ValueError('Invalid number of flaps specified. No model available.')

    # Calculation of the directivity function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 16
    D_tef = 3 * (jnp.sin(theta_flaps * jnp.pi / 180.) * jnp.cos(
        theta * jnp.pi / 180.) + jnp.cos(theta_flaps * jnp.pi / 180.) * jnp.sin(
        theta * jnp.pi / 180.) * jnp.cos(phi * jnp.pi / 180.)) ** 2

    # Strouhal number
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 19
    S_tef = f * aircraft.af_S_f / (M_0 * aircraft.af_b_f * c_0) * (1 - M_0 * jnp.cos(theta * jnp.pi / 180.))
    # Calculation of the spectral function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 17-18
    # if aircraft.af_s < 3:
    #     if S_tef < 2:
    #         F_tef = 0.0480 * S_tef
    #     elif 2 <= S_tef <= 20:
    #         F_tef = 0.1406 * S_tef ** (-0.55)
    #     else:
    #         F_tef = 216.49 * S_tef ** (-3)
    # elif aircraft.af_s == 3:
    #     if S_tef < 2:
    #         F_tef = 0.0257 * S_tef
    #     elif 2 <= S_tef <= 75:
    #         F_tef = 0.0536 * S_tef ** (-0.0625)
    #     else:
    #         F_tef = 17078 * S_tef ** (-3)
    if aircraft.af_s < 3:
        F_tef = 216.49 * S_tef ** (-3)
        F_tef[S_tef < 2] = (0.0480 * S_tef)[S_tef < 2]
        F_tef[(2 <= S_tef)*(S_tef <= 20)] = (0.1406 * S_tef ** (-0.55))[(2 <= S_tef)*(S_tef <= 20)]
    elif aircraft.af_s == 3:
        F_tef = 17078 * S_tef ** (-3)
        F_tef[S_tef < 2] = (0.0257 * S_tef)[S_tef < 2]
        F_tef[(2 <= S_tef)*(S_tef <= 75)] = (0.0536 * S_tef ** (-0.0625))[(2 <= S_tef)*(S_tef <= 75)]
    else:
        raise ValueError('Invalid number of flaps specified. No model available.')

    # Calculate msap
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_s_star_af = settings['r_0'] / aircraft.af_b_w
    msap_tef = 1. / (4 * jnp.pi * r_s_star_af ** 2) / (1 - M_0 * jnp.cos(theta * jnp.pi / 180.)) ** 4 * (Pi_star_tef * D_tef * F_tef)

    return msap_tef

def landing_gear(settings, aircraft, M_0, c_0, theta, phi, I_landing_gear, f):
    """
    Compute landing gear mean-square acoustic pressure (msap)

    :param settings: pyna settings
    :type settings
    :param aircraft: aircraft parameters
    :type aircraft
    :param M_0: ambient Mach number [-]
    :type M_0
    :param c_0: ambient speed of sound [m/s]
    :type c_0
    :param theta: polar directivity angle [deg]
    :type theta
    :param phi: azimuthal directivity angle [deg]
    :type phi
    :param I_landing_gear: landing gear deflection (0/1) [-]
    :type I_landing_gear
    :param f: 1/3rd octave frequency [Hz]
    :type f

    :return: msap_lg
    :rtype
    """

    ### ---------------- Landing-gear noise ----------------
    if I_landing_gear == 1:
        # Calculate nose-gear noise
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 29
        S_ng = f * aircraft.af_d_ng / (M_0 * c_0) * (1 - M_0 * jnp.cos(theta * jnp.pi / 180.))
        # Calculate noise power and spectral distribution function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 20-21-22-25-26-27-28
        if aircraft.af_n_ng == 1 or aircraft.af_n_ng == 2:
            Pi_star_ng_w = 4.349e-4 * M_0 ** 6 * aircraft.af_n_ng * (aircraft.af_d_ng / aircraft.af_b_w) ** 2
            Pi_star_ng_s = 2.753e-4 * M_0 ** 6 * (aircraft.af_d_ng / aircraft.af_b_w) ** 2 * (aircraft.af_l_ng / aircraft.af_d_ng)
            F_ng_w = 13.59 * S_ng ** 2 * (12.5 + S_ng ** 2) ** (-2.25)
            F_ng_s = 5.32 * S_ng ** 2 * (30 + S_ng ** 8) ** (-1)
        elif aircraft.af_n_ng == 4:
            Pi_star_ng_w = 3.414 - 4 * M_0 ** 6 * aircraft.af_n_ng * (aircraft.af_d_ng / aircraft.af_b_w) ** 2
            Pi_star_ng_s = 2.753e-4 * M_0 ** 6 * (aircraft.af_d_ng / aircraft.af_b_w) ** 2 * (aircraft.af_l_ng / aircraft.af_d_ng)
            F_ng_w = 0.0577 * S_ng ** 2 * (1 + 0.25 * S_ng ** 2) ** (-1.5)
            F_ng_s = 1.28 * S_ng ** 3 * (1.06 + S_ng ** 2) ** (-3)
        else:
            raise ValueError('Invalid number of nose landing gear systems. Specify 1/2/4.')

        # Calculate main-gear noise
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 29
        S_mg = f * aircraft.af_d_mg / (M_0 * c_0) * (1 - M_0 * jnp.cos(theta * jnp.pi / 180.))
        # Calculate noise power and spectral distribution function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 20-21-22-25-26-27-28
        if aircraft.af_n_mg == 1 or aircraft.af_n_mg == 2:
            Pi_star_mg_w = 4.349e-4 * M_0 ** 6 * aircraft.af_n_mg * (aircraft.af_d_mg / aircraft.af_b_w) ** 2
            Pi_star_mg_s = 2.753e-4 * M_0 ** 6 * (aircraft.af_d_mg / aircraft.af_b_w) ** 2 * (aircraft.af_l_ng / aircraft.af_d_mg)
            F_mg_w = 13.59 * S_mg ** 2 * (12.5 + S_mg ** 2) ** (-2.25)
            F_mg_s = 5.32 * S_mg ** 2 * (30 + S_mg ** 8) ** (-1)
        elif aircraft.af_n_mg == 4:
            Pi_star_mg_w = 3.414e-4 * M_0 ** 6 * aircraft.af_n_mg * (aircraft.af_d_mg / aircraft.af_b_w) ** 2
            Pi_star_mg_s = 2.753e-4 * M_0 ** 6 * (aircraft.af_d_mg / aircraft.af_b_w) ** 2 * (aircraft.af_l_ng / aircraft.af_d_mg)
            F_mg_w = 0.0577 * S_mg ** 2 * (1 + 0.25 * S_mg ** 2) ** (-1.5)
            F_mg_s = 1.28 * S_mg ** 3 * (1.06 + S_mg ** 2) ** (-3)
        else:
            raise ValueError('Invalid number of main landing gear systems. Specify 1/2/4.')

        # Directivity function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 23-24
        D_w = 1.5 * jnp.sin(theta * jnp.pi / 180.) ** 2
        D_s = 3 * jnp.sin(theta * jnp.pi / 180.) ** 2 * jnp.sin(phi * jnp.pi / 180.) ** 2
        # Calculate msap
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
        # If landing gear is down
        r_s_star_af = settings['r_0'] / aircraft.af_b_w
        msap_lg = 1 / (4 * jnp.pi * r_s_star_af ** 2) / (1 - M_0 * jnp.cos(theta * jnp.pi / 180.)) ** 4 * (
                                aircraft.af_N_ng * (Pi_star_ng_w * F_ng_w * D_w + Pi_star_ng_s * F_ng_s * D_s) +
                                aircraft.af_N_mg * (Pi_star_mg_w * F_mg_w * D_w + Pi_star_mg_s * F_mg_s * D_s))

    # If landing gear is up
    else:
        msap_lg = 0 * theta**0 * phi**0

    return msap_lg

def airframe_fink(theta_flaps, I_lg, M_0, c_0, rho_0, mu_0, theta, phi, f, settings, aircraft, tables):
    """
    Compute aircraft noise mean-square acoustic pressure (msap).

    :param source: pyNA component computing noise sources
    :type source: Source
    :param inputs: unscaled, dimensional input variables read via inputs[key]
    :type inputs: openmdao.vectors.default_vector.DefaultVector

    :return: msap_af
    :rtype
    """

    # Calculate msap when the aircraft is not at standstill
    if not M_0 == 0:

        # Apply HSR-era aircraft calibration levels
        if settings['aircraft_hsr_calibration']:
            # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
            supp = tables.source.aircraft.supp_af_f(theta, f).reshape(settings['n_frequency_bands'], )
        else:
            supp = jnp.ones(settings['n_frequency_bands'])

        # Add aircraft noise components
        # Normalize msap by reference pressure
        msap = jnp.zeros(settings['n_frequency_bands'])
        if 'wing' in aircraft.comp_lst:
            msap_w = trailing_edge_wing(settings, aircraft, M_0, c_0, rho_0, mu_0, theta, phi, f)
            msap = msap + msap_w * supp / settings['p_ref'] ** 2
        if 'tail_v' in aircraft.comp_lst:
            msap_v = trailing_edge_vertical_tail(settings, aircraft, M_0, c_0, rho_0, mu_0, theta, phi, f)
            msap = msap + msap_v * supp / settings['p_ref'] ** 2
        if 'tail_h' in aircraft.comp_lst:
            msap_h = trailing_edge_horizontal_tail(settings, aircraft, M_0, c_0, rho_0, mu_0, theta, phi, f)
            msap = msap + msap_h * supp / settings['p_ref'] ** 2
        if 'les' in aircraft.comp_lst:
            msap_les = leading_edge_slat(settings, aircraft, M_0, c_0, rho_0, mu_0, theta, phi, f)
            msap = msap + msap_les * supp / settings['p_ref'] ** 2
        if 'tef' in aircraft.comp_lst:
            msap_tef = trailing_edge_flap(settings, aircraft, M_0, c_0, theta, phi, theta_flaps, f)
            msap = msap + msap_tef * supp / settings['p_ref'] ** 2
        if 'lg' in aircraft.comp_lst:
            msap_lg = landing_gear(settings, aircraft, M_0, c_0, theta, phi, I_lg, f)
            msap = msap + msap_lg * supp / settings['p_ref'] ** 2

    else:
        msap = 1e-99 * (jnp.ones(settings['n_frequency_bands']) * theta_flaps) ** 0

    return msap
