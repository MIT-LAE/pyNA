function jet_mixing(settings, data, ac, n_t, idx_src, input_src)

    # Extract inputs
    V_j_star = input_src[idx_src["V_j_star"][1]:idx_src["V_j_star"][2]]
    rho_j_star = input_src[idx_src["rho_j_star"][1]:idx_src["rho_j_star"][2]]
    A_j_star = input_src[idx_src["A_j_star"][1]:idx_src["A_j_star"][2]]
    Tt_j_star = input_src[idx_src["Tt_j_star"][1]:idx_src["Tt_j_star"][2]]
    M_0 = input_src[idx_src["M_0"][1]:idx_src["M_0"][2]]
    c_0 = input_src[idx_src["c_0"][1]:idx_src["c_0"][2]]
    theta = input_src[idx_src["theta"][1]:idx_src["theta"][2]]

    # Initialize solution
    T = eltype(input_src)
    msap_jet_mixing = zeros(T, (n_t, settings.N_f))
    
    r_s_star = 0.3048/sqrt(settings.A_e)
    jet_delta = 0.

    # Calculate jet mixing
    log10Vja0 = log10.(V_j_star)

    # Calculate density exponent (omega)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table II
    # Source: Hoch - Studies of the influence of density on jet noise: extend the 
    array_1 = range(-0.45, 0.6, step=0.05)
    array_2 = [-1.0, -0.9, -0.76, -0.58, -0.41, -0.22, 0.0, 0.22, 0.5, 0.77, 1.07, 1.39, 1.74, 1.95, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    if any(x->x<-0.45, log10Vja0) || any(x->x>0.6, log10Vja0)
        throw(DomainError(x," log10(V_jet/c_0) contains value outside the interpolation range of the jet density exponent, omega."))
    end
    f_omega = PCHIPInterpolation.Interpolator(array_1, array_2)    
    omega = reshape(f_omega.(log10Vja0), (n_t, 1))
    

    # Calculate power deviation factor (P)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table III
    array_1 = range(-0.45, 0.4, step=0.05)
    array_2 = [-0.13, -0.13, -0.13, -0.13, -0.13, -0.13, -0.12, -0.1, -0.05, 0.0, 0.1, 0.21, 0.32, 0.41, 0.43, 0.41,0.31, 0.14]
    if any(x->x<-0.4, log10Vja0) || any(x->x>0.4, log10Vja0)
        throw(DomainError(x," log10(V_jet/c_0) contains value outside the interpolation range of the power deviation factor, P."))
    end
    f_log10P = PCHIPInterpolation.Interpolator(array_1, array_2)
    log10P = f_log10P.(log10Vja0)
    P_function = 10 .^log10P
    
    # Calculate acoustic power (Pi_star)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Equation 3
    K = 6.67e-5
    Pi_star = reshape(K * rho_j_star .^omega .* V_j_star.^8 .* P_function, (n_t,1))

    # Calculate directivity function (D)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table IV
    f_log10D = LinearInterpolation((data.jet_D_velocity, data.jet_D_angles), data.jet_D)
    log10D = f_log10D.(theta, log10Vja0)
    D_function = reshape(10 .^log10D, (n_t,1))
        
    # Calculate Strouhal frequency adjustment factor (xi)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table V
    # Note: added extra lines in the table for V_j_star = 0, to not have set_index
    f_xi = LinearInterpolation((data.jet_xi_velocity, data.jet_xi_angles), data.jet_xi)
    xi = reshape(f_xi.(V_j_star, theta), (n_t, 1))

    # Calculate Strouhal number (St)
    D_j_star = sqrt.(4 * A_j_star / π)  # Jet diamater [-] (rel. to sqrt(settings.A_e))
    f_star = reshape(data.f, (1, settings.N_f)) .* sqrt(settings.A_e) ./ reshape(c_0, (n_t,1))
    St = (f_star .* reshape(D_j_star, (n_t,1))) ./ reshape(xi .* (V_j_star .- M_0), (n_t,1))
    log10St = log10.(St)

    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table VI
    # Note: extended the data table temperature range ([1, 2, 2.5, 3, 3.5]) with linearly extrapolated values ([0, 1, 2, 2.5, 3, 3.5, 4, 5, 6, 7]) to avoid set_index for backward_diff
    f_log10F = LinearInterpolation((data.jet_F_angles, data.jet_F_temperature, data.jet_F_velocity, data.jet_F_strouhal), data.jet_F)
    #mlog10F_a_lg = f_log10F.(theta.*ones(1, settings.N_f), 3.5*ones(n_t, settings.N_f), log10Vja0*ones(1, settings.N_f), log10St)
    #mlog10F_b_lg = f_log10F.(theta.*ones(1, settings.N_f), 3.4*ones(n_t, settings.N_f), log10Vja0*ones(1, settings.N_f), log10St)
    #mlog10F_a_sm = f_log10F.(theta.*ones(1, settings.N_f), 1.1*ones(n_t, settings.N_f), log10Vja0*ones(1, settings.N_f), log10St)
    #mlog10F_b_sm = f_log10F.(theta.*ones(1, settings.N_f), 1.0*ones(n_t, settings.N_f), log10Vja0*ones(1, settings.N_f), log10St)    
    
    # Linear extrapolation from the data table
    mlog10F = f_log10F.(theta.*ones(1, settings.N_f), Tt_j_star.*ones(1, settings.N_f), log10Vja0 .*ones(1, settings.N_f), log10St)
    #mlog10F[findall(Tt_j_star.*ones(1, settings.N_f) .> 3.5)] = ((mlog10F_a_lg .- mlog10F_b_lg) / (0.1) .* (Tt_j_star .- 3.5) .+ mlog10F_a_lg)[findall(Tt_j_star.*ones(1, settings.N_f) .> 3.5)]
    #mlog10F[findall(Tt_j_star.*ones(1, settings.N_f) .< 1)] = ((mlog10F_a_sm .- mlog10F_b_sm) / (0.1) .* (Tt_j_star .- 1.0) .+ mlog10F_b_sm)[findall(Tt_j_star.*ones(1, settings.N_f) .< 1)]
    F_function = 10 .^(-mlog10F / 10)

    # Calculate forward velocity index (m_theta)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Table VII
    array_1 = range(0, 180, step=10)
    array_2 = [3, 1.65, 1.1, 0.5, 0.2, 0, 0, 0.1, 0.4, 1, 1.9, 3, 4.7, 7, 8.5, 8.5, 8.5, 8.5, 8.5]
    f_m_theta = PCHIPInterpolation.Interpolator(array_1, array_2)
    m_theta = f_m_theta.(theta)

    # Calculate mean-square acoustic pressure (msap)
    # Source: Zorumski report 1982 part 2. Chapter 8.4 Equation 8
    msap_j = Pi_star .* reshape(A_j_star,(n_t,1)) / (4 * π * r_s_star^2) .* D_function .* F_function ./ reshape((1 .- M_0 .* cos.(π / 180. * (theta .- jet_delta))), (n_t, 1)) .* reshape(((V_j_star .- M_0) ./ V_j_star) .^ m_theta, (n_t,1))

    # Multiply with number of engines
    msap_j = msap_j * ac.n_eng

    # Normalize msap by reference pressure
    msap_jet_mixing = msap_j/settings.p_ref^2

    return msap_jet_mixing
end

# Function 
function jet_shock(settings, data, ac, n_t, idx_src, input_src)

    # Extract inputs
    V_j_star = input_src[idx_src["V_j_star"][1]:idx_src["V_j_star"][2]]
    M_j = input_src[idx_src["M_j"][1]:idx_src["M_j"][2]]
    A_j_star = input_src[idx_src["A_j_star"][1]:idx_src["A_j_star"][2]]
    Tt_j_star = input_src[idx_src["Tt_j_star"][1]:idx_src["Tt_j_star"][2]]
    M_0 = input_src[idx_src["M_0"][1]:idx_src["M_0"][2]]
    c_0 = input_src[idx_src["c_0"][1]:idx_src["c_0"][2]]
    theta = input_src[idx_src["theta"][1]:idx_src["theta"][2]]

    # Get elements of jet Mach number vector larger than 1
    msap_jet_shock = zeros(eltype(input_src), (n_t, settings.N_f))
    
    for i in range(1, n_t, step=1)
                
        # Only shock noise if jet Mach number is larger than 1
        if M_j[i] > 1

            r_s_star = 0.3048/sqrt(settings.A_e)
            jet_delta = 0.

            # Calculate beta function
            # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 4
            beta = (M_j[i].^2 .- 1).^0.5

            # Calculate eta (exponent of the pressure ratio parameter)
            # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 5
            if beta > 1
                if Tt_j_star[i] < 1.1
                    eta = 1.
                elseif Tt_j_star[i] >= 1.1
                    eta = 2.
                end
            else
                eta = 4.
            end

            # Calculate f_star
            # Source: Zorumski report 1982 part 2. Chapter 8.5 page 8-5-1 (symbols)
            f_star = reshape(data.f, (1, settings.N_f)) .* sqrt(settings.A_e) ./ c_0[i]

            # Calculate sigma parameter
            # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 3
            sigma = 7.80 * beta .* (1 .- M_0[i] .* cos.(π / 180 .* theta[i])) .* sqrt.(A_j_star[i]) .* f_star
            log10sigma = log10.(sigma)
            
            # Calculate correlation coefficient spectrum (C-function)
            # Source: Zorumski report 1982 part 2. Chapter 8.5 Table II
            # Note: extended array for log10sigma > 2 and log10sigma < -0.7
            array_1_c = range(-2.5, 3.5, step=0.1)
            array_2_c = [0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.703,0.71,0.714,0.719,0.724,0.729,0.735,0.74,0.74,0.74,0.735,0.714,0.681,0.635,0.579,0.52,0.46,0.4,0.345,0.29,0.235,0.195,0.15,0.1,0.06,0.03,0.015,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            f_C = PCHIPInterpolation.Interpolator(array_1_c, array_2_c)
            C = f_C.(log10sigma)
                
            # Calculate W function
            # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 6-7
            b = 0.23077
            W = 0.
            for k in range(1, settings.N_shock-1, step=1)
                sum_inner = 0.
                for m in range(0,settings.N_shock - (k+1), step=1)
                    # Calculate q_km
                    q_km = 1.70 * k ./ V_j_star[i] .* (1 .- 0.06 * (m + (k + 1) / 2)) .* (1 .+ 0.7 .* V_j_star[i] .* cos.(π / 180 * theta[i]))

                    # Calculate inner sum (note: the factor b in the denominator below the sine should not be there: to get same graph as Figure 4)
                    sum_inner = sum_inner .+ sin.((b * sigma .* q_km / 2)) ./ (sigma .* q_km) .* cos.(sigma .* q_km)
                end
                
                W = W .+ sum_inner .* C.^(k.^2)
            end
            W = (4. ./ (settings.N_shock .* b)) .* W
            
            # Calculate the H function
            # Source: Zorumski report 1982 part 2. Chapter 8.5 Table III (+ linear extrapolation in logspace for log10sigma < 0; as given in SAEARP876)
            array_1_H = range(-2.5, 3.5, step=0.1)
            array_2_H = [-12.19,-11.81,-11.43,-11.05,-10.67,-10.29,-9.91,-9.53,-9.15,-8.77,-8.39,-8.01,-7.63,-7.25,-6.87,-6.49,-6.11,-5.73,-5.35,-4.97,-4.59,-4.21,-3.83,-3.45,-3.07,-2.69,-2.31,-1.94,-1.59,-1.33,-1.1,-0.94,-0.88,-0.91,-0.99,-1.09,-1.17,-1.3,-1.42,-1.55,-1.67,-1.81,-1.92,-2.06,-2.18,-2.3,-2.42,-2.54,-2.66,-2.78,-2.9,-3.02,-3.14,-3.26,-3.38,-3.5,-3.62,-3.74,-3.86,-3.98,-4.1]
            f_H = PCHIPInterpolation.Interpolator(array_1_H, array_2_H)
            log10H = f_H.(log10sigma)
            if Tt_j_star[i] < 1.1
                log10H = log10H - 0.2
            end
            
            # Calculate mean-square acoustic pressure (msap)
            # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 1
            msap_j = 1.92e-3 * A_j_star[i] / (4 * π * r_s_star^2) .* (1 .+ W) ./ (1 .- M_0[i] .* cos.(π / 180. * (theta[i] .- jet_delta))).^4 .* beta.^eta .* (10 .^log10H)
            
            # Multiply with number of engines
            msap_j = msap_j * ac.n_eng

            # Normalize msap by reference pressure
            msap_jet_shock[i,:] = msap_j/settings.p_ref^2
        end
    end
    
    return msap_jet_shock
end