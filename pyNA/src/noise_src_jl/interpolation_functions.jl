using Interpolations
using PCHIPInterpolation


## Fan 
function get_fan_interpolation_functions(settings, data)
	
	# Fan suppression
    data_angles = data.supp_fi_angles
    data_freq = data.supp_fi_freq
    data_supp = data.supp_fi
	f_supp_fi = LinearInterpolation((data_freq, data_angles), data_supp)

    data_angles = data.supp_fd_angles
    data_freq = data.supp_fd_freq
    data_supp = data.supp_fd
	f_supp_fd = LinearInterpolation((data_freq, data_angles), data_supp)
    
    # Fan inlet broadband
	if settings.fan_BB_method == "kresja"
        THET7A = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
        FIG7A = [-0.5, -1, -1.25, -1.41, -1.4, -2.2, -4.5, -8.5, -13, -18.5, -24, -30, -36, -42, -48, -54, -60,-66, -73, -66]
        f_F3IB = LinearInterpolation(THET7A, FIG7A)
    elseif settings.fan_BB_method in ["original", "allied_signal", "geae"]
        THET7A = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 180, 250]
        FIG7A = [-2, -1, 0, 0, 0, -2, -4.5, -7.5, -11, -15, -19.5, -25, -63.5, -25]
        f_F3IB = LinearInterpolation(THET7A, FIG7A)
    end

    # Fan discharge broadband
    if settings.fan_BB_method == "allied_signal"
        THET7B = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        FIG7B = [0, -29.5, -26, -22.5, -19, -15.5, -12, -8.5, -5, -3.5, -2.5, -2, -1.3, 0, -3, -7, -11, -15, -20]
        f_F3DB = LinearInterpolation(THET7B, FIG7B)
    elseif settings.fan_BB_method == "kresja"
        THET7B = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
        FIG7B = [-30, -25, -20.8, -19.5, -18.4, -16.7, -14.5, -12, -9.6, -6.9, -4.5, -1.8, -0.3, 0.5, 0.7, -1.9,-4.5, -9, -15, -9]
        f_F3DB = LinearInterpolation(THET7B, FIG7B)
    elseif settings.fan_BB_method in ["original", "geae"]
        THET7B = [0, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
        FIG7B = [-41.6, -15.8, -11.5, -8, -5, -2.7, -1.2, -0.3, 0, -2, -6, -10, -15, -20, -15]
        f_F3DB = LinearInterpolation(THET7B, FIG7B)
    end

    # Fan inlet tones
    if settings.fan_RS_method == "allied_signal"
        THT13A = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        FIG13A = [-3, -1.5, -1.5, -1.5, -1.5, -2, -3, -4, -6, -9, -12.5, -16, -19.5, -23, -26.5, -30, -33.5, -37,-40.5]
        f_F3TI = LinearInterpolation(THT13A, FIG13A)
    elseif settings.fan_RS_method == "kresja"
        THT13A = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        FIG13A = [-3, -1.5, 0, 0, 0, -1.2, -3.5, -6.8, -10.5, -15.5, -19, -25, -32, -40, -49, -59, -70, -80, -90]
        f_F3TI = LinearInterpolation(THT13A, FIG13A)
    elseif settings.fan_RS_method in ["original", "geae"]  # For original and GE large fan methods:
        THT13A = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 180, 260]
        FIG13A = [-3, -1.5, 0, 0, 0, -1.2, -3.5, -6.8, -10.5, -14.5, -19, -55, -19]
        f_F3TI = LinearInterpolation(THT13A, FIG13A)
    end

    # Fan discharge tones
    if settings.fan_RS_method == "allied_signal"
        THT13B = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        FIG13B = [-34, -30, -26, -22, -18, -14, -10.5, -6.5, -4, -1, 0, 0, 0, 0, -1, -3.5, -7, -11, -16]
        f_F3TD = LinearInterpolation(THT13B, FIG13B)
    elseif settings.fan_RS_method == "kresja"
        THT13B = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        FIG13B = [-50, -41, -33, -26, -20.6, -17.9, -14.7, -11.2, -9.3, -7.1, -4.7, -2, 0, 0.8, 1, -1.6, -4.2, -9,-15]
        f_F3TD = LinearInterpolation(THT13B, FIG13B)
    elseif settings.fan_RS_method in ["original", "geae"]
        THT13B = [0, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
        FIG13B = [-39, -15, -11, -8, -5, -3, -1, 0, 0, -2, -5.5, -9, -13, -18, -13]
        f_F3TD = LinearInterpolation(THT13B, FIG13B)
    end

    # Fan combination tones
    if settings.fan_RS_method == "original"
        # Theta correction term (F2 of Eqn 8, Figure 16), original method:
        THT16 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 180, 270]
        FIG16 = [-9.5, -8.5, -7, -5, -2, 0, 0, -3.5, -7.5, -9, -13.5, -9]
        f_F2CT = LinearInterpolation(THT16, FIG16)
    elseif settings.fan_RS_method == "allied_signal"
        # Theta correction term (F2 of Eqn 8, Figure 16), small fan method:
        THT16 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 270]
        FIG16 = [-5.5, -4.5, -3, -1.5, 0, 0, 0, 0, -2.5, -5, -6, -6.9, -7.9, -8.8, -9.8, -10.7, -11.7, -12.6,-13.6, -6]
        f_F2CT = LinearInterpolation(THT16, FIG16)
    elseif settings.fan_RS_method == "geae"
        # Theta correction term (F2 of Eqn 8, Figure 16), large fan method:
        THT16 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 180, 270]
        FIG16 = [-9.5, -8.5, -7, -5, -2, 0, 0, -3.5, -7.5, -9, -13.5, -9]
        f_F2CT = LinearInterpolation(THT16, FIG16)
    elseif settings.fan_RS_method == "kresja"
        # Theta correction term (F2 of Eqn 8, Figure 16), Krejsa method:
        THT16 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
        FIG16 = [-28, -23, -18, -13, -8, -3, 0, -1.3, -2.6, -3.9, -5.2, -6.5, -7.9, -9.4, -11, -12.7, -14.5,-16.4, -18.4]
        f_F2CT = LinearInterpolation(THT16, FIG16)
    end
	
	# Suppression factors for GE#s "Flight cleanup Turbulent Control Structure."
    # Approach or takeoff values to be applied to inlet discrete interaction tones
    # at bpf and 2bpf.  Accounts for observed in-flight tendencies.
    TCSTHA = [0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180.]
    TCSAT1 = [4.8, 4.8, 5.5, 5.5, 5.3, 5.3, 5.1, 4.4, 3.9, 2.6, 2.3, 1.8, 2.1, 1.7, 1.7, 2.6, 3.5, 3.5,3.5]
    TCSAT2 = [5.8, 5.8, 3.8, 5.3, 6.4, 3.5, 3, 2.1, 2.1, 1.1, 1.4, 0.9, 0.7, 0.7, 0.4, 0.6, 0.8, 0.8,0.8]
    TCSAA1 = [5.6, 5.6, 5.8, 4.7, 4.6, 4.9, 5.1, 2.9, 3.2, 1.6, 1.6, 1.8, 2.1, 2.4, 2.2, 2, 2.8, 2.8,2.8]
    TCSAA2 = [5.4, 5.4, 4.3, 3.4, 4.1, 2, 2.9, 1.6, 1.3, 1.5, 1.1, 1.4, 1.5, 1, 1.8, 1.6, 1.6, 1.6, 1.6]

    f_TCS_takeoff_ih1 = LinearInterpolation(TCSTHA, TCSAT1)
    f_TCS_takeoff_ih2 = LinearInterpolation(TCSTHA, TCSAT2)
    f_TCS_approach_ih1 = LinearInterpolation(TCSTHA, TCSAA1)
    f_TCS_approach_ih2 = LinearInterpolation(TCSTHA, TCSAA2)
    
    return f_supp_fi, f_supp_fd, f_F3IB, f_F3DB, f_F3TI, f_F3TD, f_F2CT, f_TCS_takeoff_ih1, f_TCS_takeoff_ih2, f_TCS_approach_ih1, f_TCS_approach_ih2
end

## Combustor
function get_core_interpolation_functions()
	# Take the D function as SAE ARP876E Table 18 and all other values which are not in the table from Zorumski report 1982 part 2. Chapter 8.2 Table II
	array_1 = range(0,180,step=10)
	array_2 = [-0.85, -0.8, -0.75, -0.7, -0.65, -0.6, -0.53, -0.46, -0.39, -0.16, 0.08, 0.31, 0.5, 0.35, 0.12,-0.19,-0.51, -0.8, -0.9]
	f_D_core = LinearInterpolation(array_1, array_2)

	# Take the S function as SAE ARP876E Table 17 and all other values which are not in the table from Zorumski report 1982 part 2. Chapter 8.2 Table III
	array_1 = range(-1.7, 2.0, step=0.1)
	array_2 = [-6.27, -5.87, -5.47, -5.07, -4.67, -4.27, -3.87, -3.47, -3.12, -2.72, -2.32, -1.99, -1.7, -1.41, -1.17, -0.97, -0.82, -0.72, -0.82, -0.97, -1.17, -1.41, -1.7, -1.99, -2.32, -2.72, -3.12, -3.47, -3.87, -4.32, -4.72, -5.22, -5.7, -6.2, -6.7, -7.2, -7.7, -8.2]
	f_S_core = LinearInterpolation(array_1, array_2)

	return f_D_core, f_S_core
end

## Jet mixing noise
function get_jet_mixing_interpolation_functions(data)
	# Source: Zorumski report 1982 part 2. Chapter 8.4 Table II
	# Source: Hoch - Studies of the influence of density on jet noise: extend the 
	array_1 = range(-0.45, 0.6, step=0.05)
	array_2 = [-1.0, -0.9, -0.76, -0.58, -0.41, -0.22, 0.0, 0.22, 0.5, 0.77, 1.07, 1.39, 1.74, 1.95, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
	f_omega_jet = LinearInterpolation(array_1, array_2)    

	# Source: Zorumski report 1982 part 2. Chapter 8.4 Table III
	array_1 = range(-0.45, 0.4, step=0.05)
	array_2 = [-0.13, -0.13, -0.13, -0.13, -0.13, -0.13, -0.12, -0.1, -0.05, 0.0, 0.1, 0.21, 0.32, 0.41, 0.43, 0.41,0.31, 0.14]
	f_log10P_jet = LinearInterpolation(array_1, array_2)

	# Source: Zorumski report 1982 part 2. Chapter 8.4 Table IV
	f_log10D_jet = LinearInterpolation((data.jet_D_velocity, data.jet_D_angles), data.jet_D)

	# Source: Zorumski report 1982 part 2. Chapter 8.4 Table V
	# Note: added extra lines in the table for V_j_star = 0, to not have set_index
	f_xi_jet = LinearInterpolation((data.jet_xi_velocity, data.jet_xi_angles), data.jet_xi)

	# Source: Zorumski report 1982 part 2. Chapter 8.4 Table VI
	# Note: extended the data table temperature range ([1, 2, 2.5, 3, 3.5]) with linearly extrapolated values ([0, 1, 2, 2.5, 3, 3.5, 4, 5, 6, 7]) to avoid set_index for backward_diff
	f_log10F_jet = LinearInterpolation((data.jet_F_angles, data.jet_F_temperature, data.jet_F_velocity, data.jet_F_strouhal), data.jet_F)

	# Source: Zorumski report 1982 part 2. Chapter 8.4 Table VII
	array_1 = range(0, 180, step=10)
	array_2 = [3, 1.65, 1.1, 0.5, 0.2, 0, 0, 0.1, 0.4, 1, 1.9, 3, 4.7, 7, 8.5, 8.5, 8.5, 8.5, 8.5]
	f_m_theta_jet = LinearInterpolation(array_1, array_2)

	return f_omega_jet, f_log10P_jet, f_log10D_jet, f_xi_jet, f_log10F_jet, f_m_theta_jet
end

## Jet shock cell noise
function get_jet_shock_interpolation_functions()
	# Source: Zorumski report 1982 part 2. Chapter 8.5 Table II
	# Note: extended array for log10sigma > 2 and log10sigma < -0.7
	array_1_c = range(-3.5, 3.5, step=0.1)
	array_2_c = [0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.71,0.714,0.719,0.724,0.729,0.735,0.74,0.74,0.74,0.735,0.714,0.681,0.635,0.579,0.52,0.46,0.4,0.345,0.29,0.235,0.195,0.15,0.1,0.06,0.03,0.015,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	f_C_jet = LinearInterpolation(array_1_c, array_2_c)

	# Source: Zorumski report 1982 part 2. Chapter 8.5 Table III (+ linear extrapolation in logspace for log10sigma < 0; as given in SAEARP876)
	array_1_H = range(-2.5, 3.5, step=0.1)
	array_2_H = [-12.19,-11.81,-11.43,-11.05,-10.67,-10.29,-9.91,-9.53,-9.15,-8.77,-8.39,-8.01,-7.63,-7.25,-6.87,-6.49,-6.11,-5.73,-5.35,-4.97,-4.59,-4.21,-3.83,-3.45,-3.07,-2.69,-2.31,-1.94,-1.59,-1.33,-1.1,-0.94,-0.88,-0.91,-0.99,-1.09,-1.17,-1.3,-1.42,-1.55,-1.67,-1.81,-1.92,-2.06,-2.18,-2.3,-2.42,-2.54,-2.66,-2.78,-2.9,-3.02,-3.14,-3.26,-3.38,-3.5,-3.62,-3.74,-3.86,-3.98,-4.1]
	f_H_jet = LinearInterpolation(array_1_H, array_2_H)

	return f_C_jet, f_H_jet
end

## Airframe suppression
function get_airframe_interpolation_functions(data)
    # HSR-era airframe suppression/calibration levels
    # Source: validation noise assessment data set of NASA STCA (Berton et al., 2019)
    # Suppression data
	f_hsr_supp = LinearInterpolation((data.supp_af_freq, data.supp_af_angles), data.supp_af)

	return f_hsr_supp
end

## Propagation
function get_propagation_interpolation_functions(data)
    # Atmospheric absorption
    f_abs = LinearInterpolation((data.abs_alt, data.abs_freq), data.abs)
    f_faddeeva_real = LinearInterpolation((data.Faddeeva_itau_im, data.Faddeeva_itau_re), data.Faddeeva_real, extrapolation_bc=Flat())
    f_faddeeva_imag = LinearInterpolation((data.Faddeeva_itau_im, data.Faddeeva_itau_re), data.Faddeeva_imag, extrapolation_bc=Flat())
    
    return f_abs, f_faddeeva_real, f_faddeeva_imag
end

## Noy table
function get_noy_interpolation_functions(data)
    f_noy = LinearInterpolation((data.noy_spl, data.noy_freq), data.noy)
    
    return f_noy
end

# A-weighting
function get_a_weighting_interpolation_functions(data)

    f_aw = LinearInterpolation(data.aw_freq, data.aw_db)

    return f_aw
end






