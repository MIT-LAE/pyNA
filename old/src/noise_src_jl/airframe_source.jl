using ReverseDiff

function trailing_edge_wing!(spl::Array, x::Union{Array, ReverseDiff.TrackedArray}, settings, af, f::Array{Float64, 1})
    
    # x = [theta_flaps, I_landing_gear, c_0, rho_0, mu_0, M_0, theta, phi, hsr_supp]
    # y = spl
    
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 5
    delta_w_star = 0.37 * (af.af_S_w / af.af_b_w^2) * (x[4] * x[6] * x[3] * af.af_S_w / (x[5] * af.af_b_w))^(-0.2)

    # Determine configuration constant and the sound power
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 7
    if af.af_clean_w == 0
        K_w = 4.464e-5
    elseif af.af_clean_w == 1
        K_w = 7.075e-6
    end

    Pi_star_w = K_w * x[6]^5 * delta_w_star

    # Determine directivity function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
    D_w = 4. * cos(x[8] * π / 180.)^2 * cos(x[7] / 2 * π / 180.)^2

    # Determine spectral distribution function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-11-12
    S_w = f * delta_w_star * af.af_b_w / (x[6] * x[3]) * (1 - x[6] * cos(x[7] * π / 180.))
    if af.af_delta_wing == 1
        F_w = 0.613 * (10 * S_w).^4 .* ((10 * S_w).^1.35 .+ 0.5).^(-4)
    elseif af.af_delta_wing == 0
        F_w = 0.485 * (10 * S_w).^4 .* ((10 * S_w).^1.5 .+ 0.5).^(-4)
    end
    
    # Add msap
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_s_star_af = settings["r_0"] / af.af_b_w
    pow_level_af = 1/settings["p_ref"]^2 / (4 * π * r_s_star_af^2) / (1 - x[6] * cos(x[7] * π / 180.))^4 * Pi_star_w * D_w

    spl .+= pow_level_af .* F_w .* x[9:end]
end
function trailing_edge_horizontal_tail!(spl::Array, x::Union{Array, ReverseDiff.TrackedArray}, settings, af, f::Array{Float64, 1})

    # x = [theta_flaps, I_landing_gear, c_0, rho_0, mu_0, M_0, theta, phi, hsr_supp]
    # y = spl
    
    # Trailing edge noise of the horizontal tail
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 5
    delta_h_star = 0.37 * (af.af_S_h / af.af_b_h^2) * (x[4] * x[6] * x[3] * af.af_S_h / (x[5] * af.af_b_h))^(-0.2)
    
    # Determine configuration constant and the sound power
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 7
    if af.af_clean_h == 0
        K_h = 4.464e-5
    elseif af.af_clean_h == 1
        K_h = 7.075e-6
    end
    Pi_star_h = K_h * x[6]^5 * delta_h_star * (af.af_b_h / af.af_b_w)^2
    
    # Determine the directivity function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
    D_h = 4 * cos(x[8] * π / 180.)^2 * cos(x[7] / 2 * π / 180.)^2
    
    # Determine spectral distribution function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-11-12
    S_h = f * delta_h_star * af.af_b_h / (x[6] * x[3]) * (1 - x[6] * cos(x[7] * π / 180.))
    F_h = 0.485 * (10 * S_h).^4 .* ((10 * S_h).^1.5 .+ 0.5).^(-4)
    
    # Add msap
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_s_star_af = settings["r_0"] / af.af_b_w
    pow_level_af = 1/settings["p_ref"]^2 / (4. * π * r_s_star_af^2) / (1 - x[6] * cos(x[7] * π / 180.))^4 * Pi_star_h * D_h

    spl .+= pow_level_af .* F_h .* x[9:end]
end
function trailing_edge_vertical_tail!(spl::Array, x::Union{Array, ReverseDiff.TrackedArray}, settings, af, f)

    # x = [theta_flaps, I_landing_gear, c_0, rho_0, mu_0, M_0, theta, phi, hsr_supp]
    # y = spl
    
    delta_v_star = 0.37 * (af.af_S_v / af.af_b_v^2) * (x[4] * x[6] * x[3] * af.af_S_v / (x[5] * af.af_b_v))^(-0.2)

    # Trailing edge noise of the vertical tail
    if af.af_clean_v == 0
        K_v = 4.464e-5
    elseif af.af_clean_v == 1
        K_v = 7.075e-6
    end
    
    Pi_star_v = K_v * x[6]^5 * delta_v_star * (af.af_b_v / af.af_b_w)^2

    # Determine directivity function
    D_v = 4 * sin(x[8] * π / 180.)^2 * cos(x[7] / 2 * π / 180.)^2

    # Determine spectral distribution function
    S_v = f * delta_v_star * af.af_b_v / (x[6] * x[3]) * (1 - x[6] * cos(x[7] * π / 180.))
    
    if af.af_delta_wing == 1
        F_v = 0.613 * (10 * S_v).^4 .* ((10 * S_v).^1.35 .+ 0.5).^(-4)
    elseif af.af_delta_wing == 0
        F_v = 0.485 * (10 * S_v).^4 .* ((10 * S_v).^1.35 .+ 0.5).^(-4)
    end
    
    # Determine msap
    r_s_star_af = settings["r_0"] / af.af_b_w
    pow_level_af = 1/settings["p_ref"]^2 / (4 * π * r_s_star_af^2) / (1 - x[6] * cos(x[7] * π / 180.))^4 * Pi_star_v * D_v 

    spl .+= pow_level_af .* F_v .* x[9:end]
end
function leading_edge_slat!(spl::Array, x::Union{Array, ReverseDiff.TrackedArray}, settings, af, f::Array{Float64,1})
    
    # x = [theta_flaps, I_landing_gear, c_0, rho_0, mu_0, M_0, theta, phi, hsr_supp]
    # y = spl
    
    # Trailing edge noise leading edge flap
    delta_w_star = 0.37 * (af.af_S_w / af.af_b_w^2) * (x[4] * x[6] * x[3] * af.af_S_w / (x[5] * af.af_b_w))^(-0.2)

    # Noise power
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 4
    Pi_star_les1 = 4.464e-5 * x[6]^5 * delta_w_star  # Slat noise
    Pi_star_les2 = 4.464e-5 * x[6]^5 * delta_w_star  # Added trailing edge noise
    
    # Determine the directivity function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 8
    D_les = 4 * cos(x[8] * π / 180.)^2 * cos(x[7] / 2 * π / 180.)^2
    
    # Determine spectral distribution function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 10-12-13
    S_les = f * delta_w_star * af.af_b_w / (x[6] * x[3]) * (1 - x[6] * cos(x[7] * π / 180.))
    
    F_les1 = 0.613 * (10 * S_les).^4 .* ((10. * S_les).^1.5 .+ 0.5).^(-4)
    F_les2 = 0.613 * (2.19 * S_les).^4 .* ((2.19 * S_les).^1.5 .+ 0.5).^(-4)
    
    # Add msap
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    r_s_star_af = settings["r_0"] / af.af_b_w
    pow_level_af_1 = 1/settings["p_ref"]^2 / (4 * π * r_s_star_af^2) / (1 - x[6] * cos(x[7] * π / 180.))^4 * Pi_star_les1 * D_les
    pow_level_af_2 = 1/settings["p_ref"]^2 / (4 * π * r_s_star_af^2) / (1 - x[6] * cos(x[7] * π / 180.))^4 * Pi_star_les2 * D_les

    spl .+= (pow_level_af_1 * F_les1 +  pow_level_af_2 * F_les2) .* x[9:end]
end
function trailing_edge_flap!(spl::Array, x::Union{Array, ReverseDiff.TrackedArray}, settings, af, f::Array{Float64,1})
    
    # x = [theta_flaps, I_landing_gear, c_0, rho_0, mu_0, M_0, theta, phi, hsr_supp]
    # y = spl
    
    # Calculate noise power
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 14-15
    if af.af_s < 3
        Pi_star_tef = 2.787e-4 * x[6]^6 * af.af_S_f / af.af_b_w^2 * sin(x[1] * π / 180.)^2
    elseif af.af_s == 3
        Pi_star_tef = 3.509e-4 * x[6]^6 * af.af_S_f / af.af_b_w^2 * sin(x[1] * π / 180.)^2
    end
        
    # Calculation of the directivity function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 16
    D_tef = 3 * (sin(x[1] * π / 180.) * cos(x[7] * π / 180.) + cos(x[1] * π / 180.) * sin(x[7] * π / 180.) * cos(x[8] * π / 180.))^2
    # Strouhal number
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 19
    S_tef = f * af.af_S_f / (x[6] * af.af_b_f * x[3]) * (1 - x[6] * cos(x[7] * π / 180.))
    
    # Calculation of the spectral function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 17-18
    if af.af_s < 3
        F_tef = zeros(eltype(S_tef), (settings["n_frequency_bands"], ))
        for i in range(1, settings["n_frequency_bands"], step=1)
            if S_tef[i] .< 2.
                F_tef[i] = 0.0480 * S_tef[i]
            elseif 2. .< S_tef[i] .< 20.
                F_tef[i] = 0.1406 * S_tef[i].^(-0.55)
            else
                F_tef[i] = 216.49 * S_tef[i].^(-3)
            end
        end
    elseif af.af_s == 3
        F_tef = zeros(eltype(S_tef), (settings["n_frequency_bands"], ))
        for i in range(1, settings["n_frequency_bands"], step=1)
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
    r_s_star_af = settings["r_0"] / af.af_b_w
    pow_level_af = 1/settings["p_ref"]^2 / (4 * π * r_s_star_af^2) / (1 - x[6] * cos(x[7] * π / 180.))^4 * Pi_star_tef * D_tef

    spl .+= pow_level_af .* F_tef .* x[9:end]
end
function landing_gear!(spl::Array, x::Union{Array, ReverseDiff.TrackedArray}, settings, af, f::Array{Float64, 1})

    # x = [theta_flaps, I_landing_gear, c_0, rho_0, mu_0, M_0, theta, phi, hsr_supp]
    # y = spl
    
    # Calculate nose-gear noise
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 29
    S_ng = f * af.af_d_ng / (x[6] * x[3]) * (1 - x[6] * cos(x[7] * π / 180.))
    
    # Calculate noise power and spectral distribution function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 20-21-22-25-26-27-28
    if af.af_n_ng == 1 || af.af_n_ng == 2
        Pi_star_ng_w = 4.349e-4 * x[6]^6 * af.af_n_ng * (af.af_d_ng / af.af_b_w)^2
        Pi_star_ng_s = 2.753e-4 * x[6]^6 * (af.af_d_ng / af.af_b_w)^2 * (af.af_l_ng / af.af_d_ng)
        F_ng_w = 13.59 * S_ng.^2 .* (12.5 .+ S_ng.^2).^(-2.25)
        F_ng_s = 5.32 * S_ng.^2 .* (30 .+ S_ng.^8).^(-1)
    elseif af.af_n_ng == 4
        Pi_star_ng_w = 3.414 - 4 * x[6]^6 * af.af_n_ng * (af.af_d_ng / af.af_b_w)^2
        Pi_star_ng_s = 2.753e-4 * x[6]^6 * (af.af_d_ng / af.af_b_w)^2 * (af.af_l_ng / af.af_d_ng)
        F_ng_w = 0.0577 * S_ng.^2 .* (1 .+ 0.25 * S_ng.^2).^(-1.5)
        F_ng_s = 1.28 * S_ng.^3 .* (1.06 .+ S_ng.^2).^(-3)
    end
    
    # Calculate main-gear noise
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 29
    S_mg = f * af.af_d_mg / (x[6] * x[3]) * (1 - x[6] * cos(x[7] * π / 180.))
    # Calculate noise power and spectral distribution function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 20-21-22-25-26-27-28
    if af.af_n_mg == 1 || af.af_n_mg == 2
        Pi_star_mg_w = 4.349e-4 * x[6]^6 * af.af_n_mg * (af.af_d_mg / af.af_b_w)^2
        Pi_star_mg_s = 2.753e-4 * x[6]^6 * (af.af_d_mg / af.af_b_w)^2 * (af.af_l_ng / af.af_d_mg)
        F_ng_w = 13.59 * S_mg.^2 .* (12.5 .+ S_mg.^2).^(-2.25)
        F_ng_s = 5.32 * S_mg.^2 .* (30 .+ S_mg.^8).^(-1)
    elseif af.af_n_mg == 4
        Pi_star_mg_w = 3.414e-4 * x[6]^6 * af.af_n_mg * (af.af_d_mg / af.af_b_w)^2
        Pi_star_mg_s = 2.753e-4 * x[6]^6 * (af.af_d_mg / af.af_b_w)^2 * (af.af_l_ng / af.af_d_mg)
        F_mg_w = 0.0577 * S_mg.^2 .* (1 .+ 0.25 * S_mg.^2).^(-1.5)
        F_mg_s = 1.28 * S_mg.^3 .* (1.06 .+ S_mg.^2).^(-3)
    end
    
    # Directivity function
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 23-24
    D_w = 1.5 * sin(x[7] * π / 180.)^2
    D_s = 3 * sin(x[7] * π / 180.)^2 * sin(x[8] * π / 180.)^2
    # Calculate msap
    # Source: Zorumski report 1982 part 2. Chapter 8.8 Equation 1
    # If landing gear is down
    r_s_star_af = settings["r_0"] / af.af_b_w
    pow_level_af_ng_w = 1/settings["p_ref"]^2 / (4 * π * r_s_star_af^2) / (1 - x[6] * cos(x[7] * π / 180.))^4 * af.af_N_ng * Pi_star_ng_w * D_w
    pow_level_af_ng_s = 1/settings["p_ref"]^2 / (4 * π * r_s_star_af^2) / (1 - x[6] * cos(x[7] * π / 180.))^4 * af.af_N_ng * Pi_star_ng_s * D_s
    pow_level_af_mg_w = 1/settings["p_ref"]^2 / (4 * π * r_s_star_af^2) / (1 - x[6] * cos(x[7] * π / 180.))^4 * af.af_N_mg * Pi_star_mg_w * D_w
    pow_level_af_mg_s = 1/settings["p_ref"]^2 / (4 * π * r_s_star_af^2) / (1 - x[6] * cos(x[7] * π / 180.))^4 * af.af_N_mg * Pi_star_mg_s * D_s

    spl .+= x[2].*(pow_level_af_ng_w * F_ng_w + pow_level_af_ng_s * F_ng_s + pow_level_af_mg_w * F_mg_w + pow_level_af_mg_s * F_mg_s) .* x[9:end]
end

function airframe_source!(spl::Array, x::Union{Array, ReverseDiff.TrackedArray}, settings, pyna_ip, af, f::Array{Float64,1})
    
    # x = [theta_flaps, I_landing_gear, c_0, rho_0, mu_0, M_0, theta, phi]
    # y = spl

    # HSR calibration
    hsr_supp = pyna_ip.f_hsr_supp.(f, x[7]*ones(eltype(x), (settings["n_frequency_bands"], )))

    # Add airframe noise components
    if "wing" in af.comp_lst
        trailing_edge_wing!(spl, vcat(x, hsr_supp), settings, af, f)
    end
    if "tail_v" in af.comp_lst
        trailing_edge_vertical_tail!(spl, vcat(x, hsr_supp), settings, af, f)
    end
    if "tail_h" in af.comp_lst
        trailing_edge_horizontal_tail!(spl, vcat(x, hsr_supp), settings, af, f)
    end
    if "les" in af.comp_lst
        leading_edge_slat!(spl, vcat(x, hsr_supp), settings, af, f)
    end
    if "tef" in af.comp_lst
        trailing_edge_flap!(spl, vcat(x, hsr_supp), settings, af, f)
    end
    if "lg" in af.comp_lst
        landing_gear!(spl, vcat(x, hsr_supp), settings, af, f)
    end

end

airframe_source_fwd! = (y,x)->airframe_source!(y,x, settings, pyna_ip, af, f)