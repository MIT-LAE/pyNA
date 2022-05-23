function core_ge(settings::PyObject, data::PyObject, ac::PyObject, n_t::Int64, M_0, theta, mdoti_c_star, Tti_c_star, Ttj_c_star, Pti_c_star, DTt_des_c_star)
    
    # Initialize solution
    T = eltype(mdoti_c_star)
    msap_core = zeros(T, (n_t, settings.N_f))
    r_s_star = settings.r_0/sqrt(settings.A_e)
    A_c_star = 1.

    # Step through all time steps
    for i in 1:n_t

        # Turbine transmission loss function
        # Source: Zorumski report 1982 part 2. Chapter 8.2 Equation 3
        g_TT = DTt_des_c_star[i]^(-4)
        
        # Calculate acoustic power (Pi_star)
        # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 3
        Pi_star = 8.85e-7 * (mdoti_c_star[i] / A_c_star) * ((Ttj_c_star[i] - Tti_c_star[i]) / Tti_c_star[i])^2 * Pti_c_star[i]^2 * g_TT
        
        # Calculate directivity function (D)
        # Take the D function as SAE ARP876E Table 18 and all other values which are not in the table from Zorumski report 1982 part 2. Chapter 8.2 Table II
        array_1 = range(0,180,step=10)
        array_2 = [-0.85, -0.8, -0.75, -0.7, -0.65, -0.6, -0.53, -0.46, -0.39, -0.16, 0.08, 0.31, 0.5, 0.35, 0.12,-0.19,-0.51, -0.8, -0.9]
        f_D = LinearInterpolation(array_1, array_2)
        D_funct = 10^f_D(theta[i])

        # Calculate the spectral function (S)
        # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 4
        f_p = 400 / (1 - M_0[i] * cos(theta[i] * π / 180.))
        log10ffp = log10.(reshape(data.f, (1, settings.N_f)) / f_p)

        # Take the S function as SAE ARP876E Table 17 and all other values which are not in the table from Zorumski report 1982 part 2. Chapter 8.2 Table III
        array_1 = range(-1.1, 1.6, step=0.1)
        array_2 = [-3.87, -3.47, -3.12, -2.72, -2.32, -1.99, -1.7, -1.41, -1.17, -0.97, -0.82, -0.72, -0.82, -0.97, -1.17, -1.41, -1.7, -1.99, -2.32, -2.72, -3.12, -3.47, -3.87, -4.32, -4.72, -5.22, -5.7, -6.2]
        f_S = LinearInterpolation(array_1, array_2)
        S_funct = 10 .^f_S.(log10ffp)
        
        # Calculate mean-square acoustic pressure (msap)
        # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 1
        # Multiply with number of engines
        # Normalize msap by reference pressure
        msap_core[i,:] = Pi_star * A_c_star / (4 * π * r_s_star^2) * D_funct * S_funct / (1 - M_0[i] * cos(π / 180. * theta[i]))^4 * ac.n_eng/settings.p_ref^2 

    end
    
    return msap_core

end

function core_pw(settings::PyObject, data::PyObject, ac::PyObject, n_t::Int64, M_0, theta, mdoti_c_star, Tti_c_star, Ttj_c_star, Pti_c_star, rho_te_c_star, c_te_c_star, rho_ti_c_star, c_ti_c_star)
    
    # Initialize solution
    T = eltype(mdoti_c_star)
    msap_core = zeros(T, (n_t, settings.N_f))
    r_s_star = settings.r_0/sqrt(settings.A_e)
    A_c_star = 1.

    # Step through all time steps
    for i in 1:n_t

        # Turbine transmission loss function
        # Source: Hultgren, 2012: A comparison of combustor models Equation 6
        zeta = (rho_te_c_star[i] * c_te_c_star[i]) / (rho_ti_c_star[i] * c_ti_c_star[i])
        g_TT = 0.8 * zeta / (1 + zeta)^2
        
        # Calculate acoustic power (Pi_star)
        # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 3
        Pi_star = 8.85e-7 * (mdoti_c_star[i] / A_c_star) * ((Ttj_c_star[i] - Tti_c_star[i]) / Tti_c_star[i])^2 * Pti_c_star[i]^2 * g_TT
        
        # Calculate directivity function (D)
        # Take the D function as SAE ARP876E Table 18 and all other values which are not in the table from Zorumski report 1982 part 2. Chapter 8.2 Table II
        array_1 = range(0,180,step=10)
        array_2 = [-0.85, -0.8, -0.75, -0.7, -0.65, -0.6, -0.53, -0.46, -0.39, -0.16, 0.08, 0.31, 0.5, 0.35, 0.12,-0.19,-0.51, -0.8, -0.9]
        #f_D = PCHIPInterpolation.Interpolator(array_1, array_2)
        f_D = LinearInterpolation(array_1, array_2)
        D_funct = 10^f_D(theta[i])

        # Calculate the spectral function (S)
        # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 4
        f_p = 400 / (1 - M_0[i] * cos(theta[i] * π / 180.))
        log10ffp = log10.(reshape(data.f, (1, settings.N_f)) / f_p)

        # Take the S function as SAE ARP876E Table 17 and all other values which are not in the table from Zorumski report 1982 part 2. Chapter 8.2 Table III
        array_1 = range(-1.1, 1.6, step=0.1)
        array_2 = [-3.87, -3.47, -3.12, -2.72, -2.32, -1.99, -1.7, -1.41, -1.17, -0.97, -0.82, -0.72, -0.82, -0.97, -1.17, -1.41, -1.7, -1.99, -2.32, -2.72, -3.12, -3.47, -3.87, -4.32, -4.72, -5.22, -5.7, -6.2]
        f_S = LinearInterpolation(array_1, array_2)
        S_funct = 10. .^f_S.(log10ffp)
        
        # Calculate mean-square acoustic pressure (msap)
        # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 1
        # Multiply with number of engines
        # Normalize msap by reference pressure
        msap_core[i,:] = Pi_star * A_c_star / (4 * π * r_s_star^2) * D_funct .* S_funct / (1. - M_0 * cos(π / 180. * theta))^4 *ac.n_eng/settings.p_ref^2 

    end

    return msap_core
    
end