function core_ge!(spl, pyna_ip, settings, ac, f, M_0, theta, TS, mdoti_c_star, Tti_c_star, Ttj_c_star, Pti_c_star, DTt_des_c_star)
    
    # Initialize solution
    r_s_star = settings.r_0/sqrt(settings.A_e)
    A_c_star = 1.

    # Turbine transmission loss function
    # Source: Zorumski report 1982 part 2. Chapter 8.2 Equation 3
    g_TT = DTt_des_c_star^(-4)
    
    # Calculate acoustic power (Pi_star)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 3
    Pi_star = 8.85e-7 * (mdoti_c_star / A_c_star) * ((Ttj_c_star - Tti_c_star) / Tti_c_star)^2 * Pti_c_star^2 * g_TT
    
    # Calculate directivity function (D)
    D_funct = 10 ^pyna_ip.f_D_core(theta)

    # Calculate the spectral function (S)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 4
    f_p = 400 / (1 - M_0 * cos.(theta * π / 180.))
    log10ffp = log10.(f / f_p)
    S_funct = 10 .^pyna_ip.f_S_core.(log10ffp)
      
    # # Calculate mean-square acoustic pressure (msap)
    # # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 1
    # # Multiply with number of engines and normalize msap by reference pressure
    pow_level_core = ac.n_eng / settings.p_ref^2 * Pi_star * A_c_star / (4 * π * r_s_star^2) * D_funct / (1 - M_0 * cos(π / 180. * theta))^4
    if settings.suppression && settings.case_name in ["nasa_stca_standard", "stca_enginedesign_standard"] && TS > 0.8
        @. spl += pow_level_core * 10 ^(-2.3 / 10.) * S_funct
    else
        @. spl += pow_level_core * S_funct
    end

end


function core_pw!(spl, pyna_ip, settings, ac, f, M_0, theta, TS, mdoti_c_star, Tti_c_star, Ttj_c_star, Pti_c_star, rho_te_c_star, c_te_c_star, rho_ti_c_star, c_ti_c_star)
    
    # Initialize solution
    r_s_star = settings.r_0/sqrt(settings.A_e)
    A_c_star = 1.

    # Turbine transmission loss function
    # Source: Hultgren, 2012: A comparison of combustor models Equation 6
    zeta = (rho_te_c_star * c_te_c_star) / (rho_ti_c_star * c_ti_c_star)
    g_TT = 0.8 * zeta / (1 + zeta)^2
    
    # Calculate acoustic power (Pi_star)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 3
    Pi_star = 8.85e-7 * (mdoti_c_star / A_c_star) * ((Ttj_c_star - Tti_c_star) / Tti_c_star)^2 * Pti_c_star^2 * g_TT
    
    # Calculate directivity function (D)
    D_funct = 10 ^pyna_ip.f_D_core(theta)

    # Calculate the spectral function (S)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 4
    f_p = 400 / (1 - M_0 * cos(theta * π / 180.))
    log10ffp = log10.(f / f_p)

    # Take the S function as SAE ARP876E Table 17 and all other values which are not in the table from Zorumski report 1982 part 2. Chapter 8.2 Table III
    S_funct = 10 .^pyna_ip.f_S_core.(log10ffp)

    # Calculate mean-square acoustic pressure (msap)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 1
    # Multiply with number of engines and normalize msap by reference pressure
    pow_level_core = ac.n_eng / settings.p_ref^2 * Pi_star * A_c_star / (4 * π * r_s_star^2) * D_funct / (1. - M_0 * cos.(π / 180. * theta))^4
    if settings.suppression && settings.case_name in ["nasa_stca_standard", "stca_enginedesign_standard"] && TS > 0.8
        @. spl += pow_level_core * 10 ^(-2.3 / 10.) * S_funct
    else
        @. spl += pow_level_core * S_funct
    end

end