function jet_mixing_source!(spl, x, settings, pyna_ip, af, f)
    
    # x = [V_j, rho_j, A_j, Tt_j, c_0, T_0, rho_0, M_0, TS, theta]
    # y = spl

    # Normalize engine variables
    V_j_star = x[1] / x[5] 
    rho_j_star = x[2] / x[7]
    A_j_star = x[3] / settings["A_e"]
    Tt_j_star = x[4] / x[6]

    r_s_star = 0.3048/sqrt(settings["A_e"])
    jet_delta = 0.

    # Calculate jet mixing
    log10Vja0 = log10(V_j_star)

    # Calculate density exponent (omega)
    omega = pyna_ip.f_omega_jet(log10Vja0)
    
    # Calculate power deviation factor (P)
    log10P = pyna_ip.f_log10P_jet(log10Vja0)
    P_function = 10^log10P
    
    # Calculate acoustic power (Pi_star)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Equation 3
    K = 6.67e-5
    Pi_star = K * rho_j_star^omega * V_j_star^8 * P_function

    # Calculate directivity function (D)
    log10D = pyna_ip.f_log10D_jet(x[10], log10Vja0)
    D_function = 10^log10D
    
    # Calculate Strouhal frequency adjustment factor (xi)
    xi = pyna_ip.f_xi_jet(V_j_star, x[10])

    # Calculate Strouhal number (St)
    D_j_star = sqrt.(4 * A_j_star / π)  # Jet diamater [-] (rel. to sqrt(settings["A_e))
    f_star = f .* sqrt(settings["A_e"]) / x[5]
    St = (f_star * D_j_star) / (xi * (V_j_star - x[8]))
    log10St = log10.(St)

    # Calculate frequency function (F)
    mlog10F = pyna_ip.f_log10F_jet.(x[10]*ones(settings["n_frequency_bands"],), Tt_j_star*ones(settings["n_frequency_bands"],), log10Vja0*ones(settings["n_frequency_bands"],), log10St)
    F_function = 10 .^(-mlog10F / 10)

    # Calculate forward velocity index (m_theta)
    m_theta = pyna_ip.f_m_theta_jet(x[10])

    # Calculate mean-square acoustic pressure (msap)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Equation 8
    # Multiply with number of engines and normalize msap by reference pressure
    pow_level_jet = af.n_eng / settings["p_ref"]^2 * Pi_star * A_j_star / (4 * π * r_s_star^2) * D_function / (1 - x[8] * cos(π / 180 * (x[10] - jet_delta))) * ((V_j_star - x[8]) / V_j_star) ^ m_theta
    if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"] && x[9] > 0.8
        spl .+= pow_level_jet * 10 ^(-2.3 / 10.) * F_function
    else
        spl .+= pow_level_jet * F_function
    end

end

function jet_shock_source!(spl, x, settings, pyna_ip, af, f)


    # x = [V_j, M_j, A_j, Tt_j, c_0, T_0, M_0, TS, theta]
    # y = spl

    # Normalize engine variables
    V_j_star = x[1] / x[5] 
    A_j_star = x[3] / settings["A_e"]
    Tt_j_star = x[4] / x[6]

    r_s_star = 0.3048/sqrt(settings["A_e"])
    jet_delta = 0.

    # Calculate beta function
    # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 4
    beta = (M_j^2 - 1)^0.5

    # Calculate eta (exponent of the pressure ratio parameter)
    # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 5
    if beta > 1
        if Tt_j_star < 1.1
            eta = 1.
        elseif Tt_j_star >= 1.1
            eta = 2.
        end
    else
        eta = 4.
    end

    # Calculate f_star
    # Source: Zorumski report 1982 part 2. Chapter 8.5 page 8-5-1 (symbols)
    f_star = f * sqrt(settings["A_e"]) / x[5]

    # Calculate sigma parameter
    # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 3
    sigma = 7.80 * beta * (1 - x[7] * cos(π / 180 * x[9])) * sqrt.(A_j_star) * f_star
    log10sigma = log10.(sigma)
    
    # Calculate correlation coefficient spectrum (C-function)
    C = pyna_ip.f_C_jet.(log10sigma)
        
    # Calculate W function
    # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 6-7
    b = 0.23077
    W = 0.
    for k in range(1, settings["n_shock"]-1, step=1)
        sum_inner = 0.
        for m in range(0,settings["n_shock"] - (k+1), step=1)
            # Calculate q_km
            q_km = 1.70 * k / V_j_star * (1 - 0.06 * (m + (k + 1) / 2)) * (1 + 0.7 * V_j_star * cos(π / 180 * x[9]))

            # Calculate inner sum (note: the factor b in the denominator below the sine should not be there: to get same graph as Figure 4)
            sum_inner = sum_inner .+ sin.((b * sigma * q_km / 2)) ./ (sigma * q_km) .* cos.(sigma * q_km)
        end
        
        W = W .+ sum_inner .* C.^(k^2)
    end
    W = (4 / (settings["n_shock"] * b)) * W
    
    # Calculate the H function
    log10H = pyna_ip.f_H_jet.(log10sigma)
    if Tt_j_star < 1.1
        log10H = log10H .- 0.2
    end

    # Calculate mean-square acoustic pressure (msap)
    # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 1
    # Multiply with number of engines and normalize msap by reference pressure
    if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"] && x[8] > 0.8
        spl .+= af.n_eng / settings["p_ref"]^2 * 1.92e-3 * A_j_star / (4 * π * r_s_star^2) * (1 + W) / (1 - x[7] * cos(π / 180 * (x[9] - jet_delta)))^4 * beta^eta * (10^log10H) * 10 ^(-2.3 / 10.)
    else
        spl .+= af.n_eng / settings["p_ref"]^2 * 1.92e-3 * A_j_star / (4 * π * r_s_star^2) * (1 + W) / (1 - x[7] * cos(π / 180 * (x[9] - jet_delta)))^4 * beta^eta * (10^log10H)
    end

end


jet_mixing_source_fwd! = (y,x)->jet_mixing_source!(y, x, settings, pyna_ip, af, f)
jet_shock_source_fwd! = (y,x)->jet_shock_source!(y, x, settings, pyna_ip, af, f)

