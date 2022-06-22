function jet_mixing!(spl, pyna_ip, settings, ac, f, M_0, c_0, theta, TS, V_j_star, rho_j_star, A_j_star, Tt_j_star)
    
    r_s_star = 0.3048/sqrt(settings.A_e)
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
    log10D = pyna_ip.f_log10D_jet(theta, log10Vja0)
    D_function = 10^log10D
    
    # Calculate Strouhal frequency adjustment factor (xi)
    xi = pyna_ip.f_xi_jet(V_j_star, theta)

    # Calculate Strouhal number (St)
    D_j_star = sqrt.(4 * A_j_star / π)  # Jet diamater [-] (rel. to sqrt(settings.A_e))
    f_star = f .* sqrt(settings.A_e) / c_0
    St = (f_star * D_j_star) / (xi * (V_j_star - M_0))
    log10St = log10.(St)

    # Calculate frequency function (F)
    mlog10F = pyna_ip.f_log10F_jet.(theta*ones(settings.N_f,), Tt_j_star*ones(settings.N_f,), log10Vja0*ones(settings.N_f,), log10St)
    F_function = 10 .^(-mlog10F / 10)

    # Calculate forward velocity index (m_theta)
    m_theta = pyna_ip.f_m_theta_jet(theta)

    # Calculate mean-square acoustic pressure (msap)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Equation 8
    # Multiply with number of engines and normalize msap by reference pressure
    pow_level_jet = ac.n_eng / settings.p_ref^2 * Pi_star * A_j_star / (4 * π * r_s_star^2) * D_function / (1 - M_0 * cos(π / 180 * (theta - jet_delta))) * ((V_j_star - M_0) / V_j_star) ^ m_theta
    if settings.suppression && settings.case_name in ["nasa_stca_standard", "stca_enginedesign_standard"] && TS > 0.8
        @. spl += pow_level_jet * 10 ^(-2.3 / 10.) * F_function
    else
        @. spl += pow_level_jet * F_function
    end

end


# Function 
function jet_shock!(spl, pyna_ip, settings, ac, f, M_0, c_0, theta, TS, V_j_star, M_j, A_j_star, Tt_j_star)

    r_s_star = 0.3048/sqrt(settings.A_e)
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
    f_star = f * sqrt(settings.A_e) / c_0

    # Calculate sigma parameter
    # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 3
    sigma = 7.80 * beta * (1 - M_0 * cos(π / 180 * theta)) * sqrt.(A_j_star) * f_star
    log10sigma = log10.(sigma)
    
    # Calculate correlation coefficient spectrum (C-function)
    C = pyna_ip.f_C_jet.(log10sigma)
        
    # Calculate W function
    # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 6-7
    b = 0.23077
    W = 0.
    for k in range(1, settings.N_shock-1, step=1)
        sum_inner = 0.
        for m in range(0,settings.N_shock - (k+1), step=1)
            # Calculate q_km
            q_km = 1.70 * k / V_j_star * (1 - 0.06 * (m + (k + 1) / 2)) * (1 + 0.7 * V_j_star * cos(π / 180 * theta))

            # Calculate inner sum (note: the factor b in the denominator below the sine should not be there: to get same graph as Figure 4)
            sum_inner = sum_inner .+ sin.((b * sigma * q_km / 2)) ./ (sigma * q_km) .* cos.(sigma * q_km)
        end
        
        W = W .+ sum_inner .* C.^(k^2)
    end
    W = (4 / (settings.N_shock * b)) * W
    
    # Calculate the H function
    log10H = pyna_ip.f_H_jet.(log10sigma)
    if Tt_j_star < 1.1
        log10H = log10H .- 0.2
    end

    # Calculate mean-square acoustic pressure (msap)
    # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 1
    # Multiply with number of engines and normalize msap by reference pressure
    if settings.suppression && settings.case_name in ["nasa_stca_standard", "stca_enginedesign_standard"] && TS > 0.8
        @. spl += ac.n_eng / settings.p_ref^2 * 1.92e-3 * A_j_star / (4 * π * r_s_star^2) * (1 + W) / (1 - M_0 * cos(π / 180 * (theta - jet_delta)))^4 * beta^eta * (10^log10H) * 10 ^(-2.3 / 10.)
    else
        @. spl += ac.n_eng / settings.p_ref^2 * 1.92e-3 * A_j_star / (4 * π * r_s_star^2) * (1 + W) / (1 - M_0 * cos(π / 180 * (theta - jet_delta)))^4 * beta^eta * (10^log10H)
    end

end