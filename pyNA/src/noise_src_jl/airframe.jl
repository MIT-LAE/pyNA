function trailing_edge_wing!(spl, settings, ac, f, M_0, c_0, rho_0, mu_0, theta, phi, hsr_supp)
    
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 5
    delta_w_star = 0.37 * (ac.af_S_w / ac.af_b_w^2) * (rho_0 * M_0 * c_0 * ac.af_S_w / (mu_0 * ac.af_b_w))^(-0.2)

    # Determine configuration constant and the sound power
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 7
    if ac.af_clean_w == 0
        K_w = 4.464e-5
    elseif ac.af_clean_w == 1
        K_w = 7.075e-6
    end

    Pi_star_w = K_w * M_0^5 * delta_w_star

    # Determine directivity function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
    D_w = 4. * cos(phi * π / 180.)^2 * cos(theta / 2 * π / 180.)^2

    # Determine spectral distribution function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-11-12
    S_w = f * delta_w_star * ac.af_b_w / (M_0 * c_0) * (1 - M_0 * cos(theta * π / 180.))
    if ac.af_delta_wing == 1
        F_w = 0.613 * (10 * S_w).^4 .* ((10 * S_w).^1.35 .+ 0.5).^(-4)
    elseif ac.af_delta_wing == 0
        F_w = 0.485 * (10 * S_w).^4 .* ((10 * S_w).^1.5 .+ 0.5).^(-4)
    end
    
    # Add msap
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_s_star_af = settings.r_0 / ac.af_b_w

    if settings.hsr_calibration
        @. spl += 1/settings.p_ref^2 / (4 * π * r_s_star_af^2) / (1 - M_0 * cos(theta * π / 180.))^4 * (Pi_star_w * D_w * F_w) * hsr_supp
    else
        @. spl += 1/settings.p_ref^2 / (4 * π * r_s_star_af^2) / (1 - M_0 * cos(theta * π / 180.))^4 * (Pi_star_w * D_w * F_w)
    end

end
function trailing_edge_horizontal_tail!(spl, settings, ac, f, M_0, c_0, rho_0, mu_0, theta, phi, hsr_supp)

    # Trailing edge noise of the horizontal tail
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 5
    delta_h_star = 0.37 * (ac.af_S_h / ac.af_b_h^2) * (rho_0 * M_0 * c_0 * ac.af_S_h / (mu_0 * ac.af_b_h))^(-0.2)
    
    # Determine configuration constant and the sound power
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 7
    if ac.af_clean_h == 0
        K_h = 4.464e-5
    elseif ac.af_clean_h == 1
        K_h = 7.075e-6
    end
    Pi_star_h = K_h * M_0^5 * delta_h_star * (ac.af_b_h / ac.af_b_w)^2
    
    # Determine the directivity function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
    D_h = 4 * cos(phi * π / 180.)^2 * cos(theta / 2 * π / 180.)^2
    
    # Determine spectral distribution function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-11-12
    S_h = f * delta_h_star * ac.af_b_h / (M_0 * c_0) * (1 - M_0 * cos(theta * π / 180.))
    F_h = 0.485 * (10 * S_h).^4 .* ((10 * S_h).^1.5 .+ 0.5).^(-4)
    
    # Add msap
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_s_star_af = settings.r_0 / ac.af_b_w
    if settings.hsr_calibration
        @. spl += 1/settings.p_ref^2 / (4. * π * r_s_star_af^2) / (1 - M_0 * cos(theta * π / 180.))^4 * (Pi_star_h * D_h * F_h) * hsr_supp
    else
        @. spl += 1/settings.p_ref^2 / (4. * π * r_s_star_af^2) / (1 - M_0 * cos(theta * π / 180.))^4 * (Pi_star_h * D_h * F_h)
    end
end
function trailing_edge_vertical_tail!(spl, settings, ac, f, M_0, c_0, rho_0, mu_0, theta, phi, hsr_supp)

    delta_v_star = 0.37 * (ac.af_S_v / ac.af_b_v^2) * (rho_0 * M_0 * c_0 * ac.af_S_v / (mu_0 * ac.af_b_v))^(-0.2)

    # Trailing edge noise of the vertical tail
    if ac.af_clean_v == 0
        K_v = 4.464e-5
    elseif ac.af_clean_v == 1
        K_v = 7.075e-6
    end
    
    Pi_star_v = K_v * M_0^5 * delta_v_star * (ac.af_b_v / ac.af_b_w)^2

    # Determine directivity function
    D_v = 4 * sin(phi * π / 180.)^2 * cos(theta / 2 * π / 180.)^2

    # Determine spectral distribution function
    S_v = f * delta_v_star * ac.af_b_v / (M_0 * c_0) * (1 - M_0 * cos(theta * π / 180.))
    
    if ac.af_delta_wing == 1
        F_v = 0.613 * (10 * S_v).^4 .* ((10 * S_v).^1.35 .+ 0.5).^(-4)
    elseif ac.af_delta_wing == 0
        F_v = 0.485 * (10 * S_v).^4 .* ((10 * S_v).^1.35 .+ 0.5).^(-4)
    end
    
    # Determine msap
    r_s_star_af = settings.r_0 / ac.af_b_w
    if settings.hsr_calibration
        @. spl += 1/settings.p_ref^2 / (4 * π * r_s_star_af^2) / (1 - M_0 * cos(theta * π / 180.))^4 * (Pi_star_v * D_v * F_v) * hsr_supp
    else
        @. spl += 1/settings.p_ref^2 / (4 * π * r_s_star_af^2) / (1 - M_0 * cos(theta * π / 180.))^4 * (Pi_star_v * D_v * F_v)
    end
end
function leading_edge_slat!(spl, settings, ac, f, M_0, c_0, rho_0, mu_0, theta, phi, hsr_supp)

    # Trailing edge noise leading edge flap
    delta_w_star = 0.37 * (ac.af_S_w / ac.af_b_w^2) * (rho_0 * M_0 * c_0 * ac.af_S_w / (mu_0 * ac.af_b_w))^(-0.2)

    # Noise power
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 4
    Pi_star_les1 = 4.464e-5 * M_0^5 * delta_w_star  # Slat noise
    Pi_star_les2 = 4.464e-5 * M_0^5 * delta_w_star  # Added trailing edge noise
    
    # Determine the directivity function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
    D_les = 4 * cos(phi * π / 180.)^2 * cos(theta / 2 * π / 180.)^2
    
    # Determine spectral distribution function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-12-13
    S_les = f * delta_w_star * ac.af_b_w / (M_0 * c_0) * (1 - M_0 * cos(theta * π / 180.))
    
    F_les1 = 0.613 * (10 * S_les).^4 .* ((10. * S_les).^1.5 .+ 0.5).^(-4)
    F_les2 = 0.613 * (2.19 * S_les).^4 .* ((2.19 * S_les).^1.5 .+ 0.5).^(-4)
    
    # Add msap
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_s_star_af = settings.r_0 / ac.af_b_w
    if settings.hsr_calibration
        @. spl += 1/settings.p_ref^2 / (4 * π * r_s_star_af^2) / (1 - M_0 * cos(theta * π / 180.))^4 * (Pi_star_les1 * D_les * F_les1 + Pi_star_les2 * D_les * F_les2) * hsr_supp
    else
        @. spl += 1/settings.p_ref^2 / (4 * π * r_s_star_af^2) / (1 - M_0 * cos(theta * π / 180.))^4 * (Pi_star_les1 * D_les * F_les1 + Pi_star_les2 * D_les * F_les2)
    end
end
function trailing_edge_flap!(spl, settings, ac, f, M_0, c_0, theta, phi, theta_flaps, hsr_supp)

    # Calculate noise power
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 14-15
    if ac.af_s < 3
        Pi_star_tef = 2.787e-4 * M_0^6 * ac.af_S_f / ac.af_b_w^2 * sin(theta_flaps * π / 180.)^2
    elseif ac.af_s == 3
        Pi_star_tef = 3.509e-4 * M_0^6 * ac.af_S_f / ac.af_b_w^2 * sin(theta_flaps * π / 180.)^2
    end
        
    # Calculation of the directivity function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 16
    D_tef = 3 * (sin(theta_flaps * π / 180.) * cos(theta * π / 180.) + cos(theta_flaps * π / 180.) * sin(theta * π / 180.) * cos(phi * π / 180.))^2
    # Strouhal number
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 19
    S_tef = f * ac.af_S_f / (M_0 * ac.af_b_f * c_0) * (1 - M_0 * cos(theta * π / 180.))
    
    # Calculation of the spectral function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 17-18
    if ac.af_s < 3
        F_tef = zeros(eltype(S_tef), (settings.N_f, ))
        for i in range(1, settings.N_f, step=1)
            if S_tef[i] .< 2.
                F_tef[i] = 0.0480 * S_tef[i]
            elseif 2. .< S_tef[i] .< 20.
                F_tef[i] = 0.1406 * S_tef[i].^(-0.55)
            else
                F_tef[i] = 216.49 * S_tef[i].^(-3)
            end
        end
    elseif ac.af_s == 3
        F_tef = zeros(eltype(S_tef), (settings.N_f, ))
        for i in range(1, settings.N_f, step=1)
            if S_tef[i] .< 2.
                F_tef[i] = 0.0257 * S_tef[i]
            elseif 2. .< S_tef[i] .< 20.
                F_tef[i] = 0.0536 * S_tef[i].^(-0.0625)
            else
                F_tef[i] = 17078 * S_tef[i].^(-3)
            end
        end
    end
        
    # Calculate msap
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_s_star_af = settings.r_0 / ac.af_b_w
    if settings.hsr_calibration
        @. spl += 1/settings.p_ref^2 / (4 * π * r_s_star_af^2) / (1 - M_0 * cos(theta * π / 180.))^4 * (Pi_star_tef * D_tef * F_tef) * hsr_supp
    else
        @. spl += 1/settings.p_ref^2 / (4 * π * r_s_star_af^2) / (1 - M_0 * cos(theta * π / 180.))^4 * (Pi_star_tef * D_tef * F_tef)
    end
end
function landing_gear!(spl, settings, ac, f, M_0, c_0, theta, phi, I_landing_gear, hsr_supp)

    if I_landing_gear == 1
        # Calculate nose-gear noise
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 29
        S_ng = f * ac.af_d_ng / (M_0 * c_0) * (1 - M_0 * cos(theta * π / 180.))
        
        # Calculate noise power and spectral distribution function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 20-21-22-25-26-27-28
        if ac.af_n_ng == 1 || ac.af_n_ng == 2
            Pi_star_ng_w = 4.349e-4 * M_0^6 * ac.af_n_ng * (ac.af_d_ng / ac.af_b_w)^2
            Pi_star_ng_s = 2.753e-4 * M_0^6 * (ac.af_d_ng / ac.af_b_w)^2 * (ac.af_l_ng / ac.af_d_ng)
            F_ng_w = 13.59 * S_ng.^2 .* (12.5 .+ S_ng.^2).^(-2.25)
            F_ng_s = 5.32 * S_ng.^2 .* (30 .+ S_ng.^8).^(-1)
        elseif ac.af_n_ng == 4
            Pi_star_ng_w = 3.414 - 4 * M_0^6 * ac.af_n_ng * (ac.af_d_ng / ac.af_b_w)^2
            Pi_star_ng_s = 2.753e-4 * M_0^6 * (ac.af_d_ng / ac.af_b_w)^2 * (ac.af_l_ng / ac.af_d_ng)
            F_ng_w = 0.0577 * S_ng.^2 .* (1 .+ 0.25 * S_ng.^2).^(-1.5)
            F_ng_s = 1.28 * S_ng.^3 .* (1.06 .+ S_ng.^2).^(-3)
        end
        
        # Calculate main-gear noise
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 29
        S_mg = f * ac.af_d_mg / (M_0 * c_0) * (1 - M_0 * cos(theta * π / 180.))
        # Calculate noise power and spectral distribution function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 20-21-22-25-26-27-28
        if ac.af_n_mg == 1 || ac.af_n_mg == 2
            Pi_star_mg_w = 4.349e-4 * M_0^6 * ac.af_n_mg * (ac.af_d_mg / ac.af_b_w)^2
            Pi_star_mg_s = 2.753e-4 * M_0^6 * (ac.af_d_mg / ac.af_b_w)^2 * (ac.af_l_ng / ac.af_d_mg)
            F_ng_w = 13.59 * S_mg.^2 .* (12.5 .+ S_mg.^2).^(-2.25)
            F_ng_s = 5.32 * S_mg.^2 .* (30 .+ S_mg.^8).^(-1)
        elseif ac.af_n_mg == 4
            Pi_star_mg_w = 3.414e-4 * M_0^6 * ac.af_n_mg * (ac.af_d_mg / ac.af_b_w)^2
            Pi_star_mg_s = 2.753e-4 * M_0^6 * (ac.af_d_mg / ac.af_b_w)^2 * (ac.af_l_ng / ac.af_d_mg)
            F_mg_w = 0.0577 * S_mg.^2 .* (1 .+ 0.25 * S_mg.^2).^(-1.5)
            F_mg_s = 1.28 * S_mg.^3 .* (1.06 .+ S_mg.^2).^(-3)
        end
        
        # Directivity function
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 23-24
        D_w = 1.5 * sin(theta * π / 180.)^2
        D_s = 3 * sin(theta * π / 180.)^2 * sin(phi * π / 180.)^2
        # Calculate msap
        # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
        # If landing gear is down
        r_s_star_af = settings.r_0 / ac.af_b_w

        if settings.hsr_calibration            
            @. spl += 1/settings.p_ref^2 / (4 * π * r_s_star_af^2) / (1 - M_0 * cos(theta * π / 180.))^4 * (ac.af_N_ng * (Pi_star_ng_w * F_ng_w * D_w + Pi_star_ng_s * F_ng_s * D_s) + ac.af_N_mg * (Pi_star_mg_w * F_mg_w * D_w + Pi_star_mg_s * F_mg_s * D_s)) * hsr_supp
        else
            @. spl += 1/settings.p_ref^2 / (4 * π * r_s_star_af^2) / (1 - M_0 * cos(theta * π / 180.))^4 * (ac.af_N_ng * (Pi_star_ng_w * F_ng_w * D_w + Pi_star_ng_s * F_ng_s * D_s) + ac.af_N_mg * (Pi_star_mg_w * F_mg_w * D_w + Pi_star_mg_s * F_mg_s * D_s))
        end
    end
end

function airframe!(spl, pyna_ip, settings, ac, f, M_0, mu_0, c_0, rho_0, theta, phi, theta_flaps, I_landing_gear)
    
    
    # HSR calibration
    hsr_supp = pyna_ip.f_hsr_supp.(f, theta*ones(eltype(theta_flaps), (settings.N_f, )))

    # Add airframe noise components
    if "wing" in ac.comp_lst
        trailing_edge_wing!(spl, settings, ac, f, M_0, c_0, rho_0, mu_0, theta, phi, hsr_supp)
    end
    if "tail_v" in ac.comp_lst
        trailing_edge_vertical_tail!(spl, settings, ac, f, M_0, c_0, rho_0, mu_0, theta, phi, hsr_supp)
    end
    if "tail_h" in ac.comp_lst
        trailing_edge_horizontal_tail!(spl, settings, ac, f, M_0, c_0, rho_0, mu_0, theta, phi, hsr_supp)
    end
    if "les" in ac.comp_lst
        leading_edge_slat!(spl, settings, ac, f, M_0, c_0, rho_0, mu_0, theta, phi, hsr_supp)
    end
    if "tef" in ac.comp_lst
        trailing_edge_flap!(spl, settings, ac, f, M_0, c_0, theta, phi, theta_flaps, hsr_supp)
    end
    if "lg" in ac.comp_lst
        landing_gear!(spl, settings, ac, f, M_0, c_0, theta, phi, I_landing_gear, hsr_supp)
    end

end