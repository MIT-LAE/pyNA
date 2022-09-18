using OpenMDAO: AbstractExplicitComp, VarData, PartialsData, get_rows_cols
import OpenMDAO: setup, compute!, compute_partials!
using Interpolations
using ReverseDiff
using LinearAlgebra
using ConcreteStructs
using PCHIPInterpolation
using Dates
using CSV
using NPZ
using DataFrames
using FLoops
using BenchmarkTools

include("get_subband_frequencies.jl")
include("get_interpolated_input_v.jl")
include("shielding.jl")
include("geometry.jl")
include("fan_source.jl")
include("core_source.jl")
include("jet_source.jl")
include("airframe_source.jl")
include("propagation.jl")
include("split_subbands.jl")
include("lateral_attenuation.jl")
include("spl.jl")
include("aspl.jl")
include("oaspl.jl")
include("pnlt.jl")
include("epnl.jl")
include("ilevel.jl")
include("aspl.jl")
include("smooth_max.jl")

function get_noise!(output_v, input_v, settings, pyna_ip, af, data, sealevel_atmosphere, n_t, n_t_noise)
    
    # Number of observers
    n_obs = size(settings["x_observer_array"])[1]
    
    # Initialize outputs
    f = 10 .^ (0.1 * (17:1:40))
    f_sb = zeros(Float64, settings["n_frequency_subbands"] * settings["n_frequency_bands"])
    
    shield = zeros(settings["n_frequency_bands"])
    spl_j = zeros(eltype(input_v), settings["n_frequency_bands"])

    t_o = zeros(eltype(input_v), (n_obs, n_t_noise))
    spl = zeros(eltype(input_v), (n_obs, n_t_noise, settings["n_frequency_bands"]))
    level = zeros(eltype(input_v), (n_obs, n_t_noise))
    level_int = zeros(eltype(input_v), n_obs)
    geom_v = zeros(eltype(input_v), 6)
    

    # 1/3rd octave band frequencies
    get_subband_frequencies!(f_sb, f, settings)

    # Interpolate the trajectory and engine inputs at reduced grid size for EPNL calculations 
    input_v_i = zeros(eltype(input_v), Int64(size(input_v)[1]/n_t*n_t_noise))
    get_interpolated_input_v!(input_v_i, input_v, settings, n_t, n_t_noise)        # 274 allocations
    
    # Iterate over observers
    for i in 1:1:n_obs
        
        println("Computing noise at observer: ", i)

        # Iterate over time steps
        for j in 1:1:n_t_noise

            # Extract inputs
            x = input_v_i[0 * n_t_noise + j]
            y = input_v_i[1 * n_t_noise + j]
            z = input_v_i[2 * n_t_noise + j]
            alpha = input_v_i[3 * n_t_noise + j]
            gamma = input_v_i[4 * n_t_noise + j]
            t_s = input_v_i[5 * n_t_noise + j]
            M_0 = input_v_i[6 * n_t_noise + j]
            n = 7
            if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
                TS = input_v_i[(n + 0) * n_t_noise + j]
                n += 1
            end
            if settings["atmosphere_type"] == "stratified"
                c_0 = input_v_i[(n + 0) * n_t_noise + j]
                T_0 = input_v_i[(n + 1) * n_t_noise + j]
                rho_0 = input_v_i[(n + 2) * n_t_noise + j]
                p_0 = input_v_i[(n + 3) * n_t_noise + j]
                mu_0 = input_v_i[(n + 4) * n_t_noise + j]
                I_0 = input_v_i[(n + 5) * n_t_noise + j]
                n += 6
            else
                c_0 = sealevel_atmosphere["c_0"]
                T_0 = sealevel_atmosphere["T_0"]
                rho_0 = sealevel_atmosphere["rho_0"]
                p_0 = sealevel_atmosphere["p_0"]
                mu_0 = sealevel_atmosphere["mu_0"]
                I_0 = sealevel_atmosphere["I_0"]
            end  
            if settings["fan_inlet_source"]==true || settings["fan_discharge_source"]==true
                DTt_f = input_v_i[(n + 0) * n_t_noise + j]
                mdot_f = input_v_i[(n + 1) * n_t_noise + j]
                N_f = input_v_i[(n + 2) * n_t_noise + j]
                A_f = input_v_i[(n + 3) * n_t_noise + j]
                d_f = input_v_i[(n + 4) * n_t_noise + j]
                n += 5
            end
            if settings["core_source"]
                if settings["core_turbine_attenuation_method"] == "ge"
                    mdoti_c = input_v_i[(n + 0) * n_t_noise + j]
                    Tti_c = input_v_i[(n + 1) * n_t_noise + j]
                    Ttj_c = input_v_i[(n + 2) * n_t_noise + j]
                    Pti_c = input_v_i[(n + 3) * n_t_noise + j]
                    DTt_des_c = input_v_i[(n + 4) * n_t_noise + j]
                    n += 5
                elseif settings["core_turbine_attenuation_method"] == "pw"
                    mdoti_c = input_v_i[(n + 0) * n_t_noise + j]
                    Tti_c = input_v_i[(n + 1) * n_t_noise + j]
                    Ttj_c = input_v_i[(n + 2) * n_t_noise + j]
                    Pti_c = input_v_i[(n + 3) * n_t_noise + j]
                    rho_te_c = input_v_i[(n + 4) * n_t_noise + j]
                    c_te_c = input_v_i[(n + 5) * n_t_noise + j]
                    rho_ti_c = input_v_i[(n + 6) * n_t_noise + j]
                    c_ti_c = input_v_i[(n + 7) * n_t_noise + j]
                    n += 8
                end
            end
            if settings["jet_mixing_source"] == true && settings["jet_shock_source"] == false
                V_j = input_v_i[(n + 0) * n_t_noise + j]
                rho_j = input_v_i[(n + 1) * n_t_noise + j]
                A_j = input_v_i[(n + 2) * n_t_noise + j]
                Tt_j = input_v_i[(n + 3) * n_t_noise + j]
                n += 4
            elseif settings["jet_shock_source"] == true && settings["jet_mixing_source"] == false
                V_j = input_v_i[(n + 0) * n_t_noise + j]
                A_j = input_v_i[(n + 1) * n_t_noise + j]
                Tt_j = input_v_i[(n + 2) * n_t_noise + j]
                M_j = input_v_i[(n + 3) * n_t_noise + j]
                n += 4
            elseif settings["jet_shock_source"] ==true && settings["jet_mixing_source"] == true
                V_j = input_v_i[(n + 0) * n_t_noise + j]
                rho_j = input_v_i[(n + 1) * n_t_noise + j]
                A_j = input_v_i[(n + 2) * n_t_noise + j]
                Tt_j = input_v_i[(n + 3) * n_t_noise + j]
                M_j = input_v_i[(n + 4) * n_t_noise + j]
                n += 5
            end
            if settings["airframe_source"]==true
                theta_flaps = input_v_i[(n + 0) * n_t_noise + j]
                I_landing_gear = input_v_i[(n + 1) * n_t_noise + j]
                n += 2
            end

            # Compute geometry: [r, beta, theta, phi, c_bar, t_o]
            geometry!(geom_v, vcat(x, y, z, alpha, gamma, t_s, c_0, T_0), settings["x_observer_array"][i,:])
            t_o[i,j] = geom_v[6]

            # Compute shielding
            if settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"] && settings["shielding"] == true
                shield .= shielding(settings, data, j, i)
            end
            
            # Compute source
            if settings["fan_inlet_source"]
                fan_source!(spl_j, vcat(DTt_f, mdot_f, N_f, A_f, d_f, c_0, rho_0, M_0, geom_v[3]), settings, pyna_ip, af, f, shield, "fan_inlet")
            end
            if settings["fan_discharge_source"]
                fan_source!(spl_j, vcat(DTt_f, mdot_f, N_f, A_f, d_f, c_0, rho_0, M_0, geom_v[3]), settings, pyna_ip, af, f, shield, "fan_discharge")
            end
            if settings["core_source"]
                if settings["core_turbine_attenuation_method"] == "ge"
                    if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
                        core_source_ge!(spl_j, vcat(mdoti_c, Tti_c, Ttj_c, Pti_c, DTt_des_c, c_0, rho_0, T_0, p_0, M_0, TS, geom_v[3]), settings, pyna_ip, af, f)
                    else
                        core_source_ge!(spl_j, vcat(mdoti_c, Tti_c, Ttj_c, Pti_c, DTt_des_c, c_0, rho_0, T_0, p_0, M_0, 1., geom_v[3]), settings, pyna_ip, af, f)
                    end
                elseif settings["core_turbine_attenuation_method"] == "pw"
                    if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
                        core_source_pw!(spl_j, vcat(mdoti_c, Tti_c, Ttj_c, Pti_c, rho_te_c, c_te_c, rho_ti_c, c_ti_c, c_0, T_0, rho_0, p_0, M_0, TS, geom_v[3]), settings, pyna_ip, af, f)
                    else
                        core_source_pw!(spl_j, vcat(mdoti_c, Tti_c, Ttj_c, Pti_c, rho_te_c, c_te_c, rho_ti_c, c_ti_c, c_0, T_0, rho_0, p_0, M_0, 1., geom_v[3]), settings, pyna_ip, af, f)
                    end
                end
            end
            if settings["jet_mixing_source"]
                if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
                    jet_mixing_source!(spl_j, vcat(V_j, rho_j, A_j, Tt_j, c_0, T_0, rho_0, M_0, TS, geom_v[3]), settings, pyna_ip, af, f)
                else
                    jet_mixing_source!(spl_j, vcat(V_j, rho_j, A_j, Tt_j, c_0, T_0, rho_0, M_0, 1., geom_v[3]), settings, pyna_ip, af, f)
                end
            end
            if settings["jet_shock_source"]
                # Only shock noise if jet Mach number is larger than 1. Choose 1.01 to avoid ill-defined derivatives.
                if M_j > 1.01
                    if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
                        jet_shock_source!(spl_j, vcat(V_j, M_j, A_j, Tt_j, c_0, T_0, M_0, TS, geom_v[3]), settings, pyna_ip, af, f)
                    else
                        jet_shock_source!(spl_j, vcat(V_j, M_j, A_j, Tt_j, c_0, T_0, M_0, 1., geom_v[3]), settings, pyna_ip, af, f)
                    end
                end
            end
            if settings["airframe_source"]
                if M_0 > 0
                    airframe_source!(spl_j, vcat(theta_flaps, I_landing_gear, c_0, rho_0, mu_0, M_0, geom_v[3], geom_v[4]), settings, pyna_ip, af, f)
                end
            end

            # Compute noise propagation
            propagation!(spl_j, vcat(geom_v[1], z, geom_v[5], rho_0, I_0, geom_v[2]), settings, pyna_ip, f_sb, settings["x_observer_array"][i,:])

            # Add spl ambient correction
            f_spl!(spl_j, vcat(c_0, rho_0))
            spl[i,j,:] = spl_j
            
            # Compute noise levels
            if settings["levels_int_metric"] == "ioaspl"
                level[i,j] = f_oaspl(spl_j)
            elseif settings["levels_int_metric"] in ["pnlt_max", "ipnlt", "epnl"]
                level[i,j] = f_pnlt(spl_j, settings, pyna_ip, f)                
            elseif settings["levels_int_metric"] == "sel"
                level[i,j] = f_aspl(spl_j, pyna_ip, f)
            end

        end

        # Compute integrated levels
        if settings["levels_int_metric"] in ["ioaspl", "ipnlt", "sel"]
            level_int[i] = f_ilevel(vcat(t_o[i,:], level[i,:]), settings)
        elseif settings["levels_int_metric"] == "epnl"
            level_int[i] = f_epnl(vcat(t_o[i,:], level[i,:]), settings)
        elseif settings["levels_int_metric"] == "pnlt_max"
            level_int[i] = smooth_max(level[i,:], 50.)
        end

        # Lateral attenuation post-hoc subtraction on integrated noise levels
        if settings["lateral_attenuation"]
            x = input_v_i[0 * n_t_noise + 1: 1 * n_t_noise]
            y = input_v_i[1 * n_t_noise + 1: 2 * n_t_noise]
            z = input_v_i[2 * n_t_noise + 1: 3 * n_t_noise]
            level_int[i] += lateral_attenuation(vcat(x, y, z), settings, settings["x_observer_array"][i,:])
        end

    end

    # Compute output_v: [level_int_lateral, level_int_flyover]
    level_lateral = smooth_max(level_int[1:end-1], 50.)
    output_v .= [level_lateral, level_int[end]]

end


function get_noise(input_v, settings, pyna_ip, af, data, sealevel_atmosphere, n_t, n_t_noise)
    
    # Number of observers
    n_obs = size(settings["x_observer_array"])[1]
    
    # Initialize outputs
    f = 10 .^ (0.1 * (17:1:40))
    f_sb = zeros(Float64, settings["n_frequency_subbands"] * settings["n_frequency_bands"])
    
    shield = zeros(settings["n_frequency_bands"])
    spl_j = zeros(eltype(input_v), settings["n_frequency_bands"])

    t_o = zeros(eltype(input_v), (n_obs, n_t_noise))
    spl = zeros(eltype(input_v), (n_obs, n_t_noise, settings["n_frequency_bands"]))
    level = zeros(eltype(input_v), (n_obs, n_t_noise))
    level_int = zeros(eltype(input_v), n_obs)
    geom_v = zeros(eltype(input_v), 6)
    
    # 1/3rd octave band frequencies
    get_subband_frequencies!(f_sb, f, settings)

    # Interpolate the trajectory and engine inputs at reduced grid size for EPNL calculations 
    input_v_i = zeros(eltype(input_v), Int64(size(input_v)[1]/n_t*n_t_noise))
    get_interpolated_input_v!(input_v_i, input_v, settings, n_t, n_t_noise)        # 274 allocations
    


    # Iterate over observers
    for i in 1:1:n_obs
        
        println("Computing noise at observer: ", i)

        # Iterate over time steps
        for j in 1:1:n_t_noise

            # Extract inputs
            x = input_v_i[0 * n_t_noise + j]
            y = input_v_i[1 * n_t_noise + j]
            z = input_v_i[2 * n_t_noise + j]
            alpha = input_v_i[3 * n_t_noise + j]
            gamma = input_v_i[4 * n_t_noise + j]
            t_s = input_v_i[5 * n_t_noise + j]
            M_0 = input_v_i[6 * n_t_noise + j]
            n = 7
            if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
                TS = input_v_i[(n + 0) * n_t_noise + j]
                n += 1
            end
            if settings["atmosphere_type"] == "stratified"
                c_0 = input_v_i[(n + 0) * n_t_noise + j]
                T_0 = input_v_i[(n + 1) * n_t_noise + j]
                rho_0 = input_v_i[(n + 2) * n_t_noise + j]
                p_0 = input_v_i[(n + 3) * n_t_noise + j]
                mu_0 = input_v_i[(n + 4) * n_t_noise + j]
                I_0 = input_v_i[(n + 5) * n_t_noise + j]
                n += 6
            else
                c_0 = sealevel_atmosphere["c_0"]
                T_0 = sealevel_atmosphere["T_0"]
                rho_0 = sealevel_atmosphere["rho_0"]
                p_0 = sealevel_atmosphere["p_0"]
                mu_0 = sealevel_atmosphere["mu_0"]
                I_0 = sealevel_atmosphere["I_0"]
            end  
            if settings["fan_inlet_source"]==true || settings["fan_discharge_source"]==true
                DTt_f = input_v_i[(n + 0) * n_t_noise + j]
                mdot_f = input_v_i[(n + 1) * n_t_noise + j]
                N_f = input_v_i[(n + 2) * n_t_noise + j]
                A_f = input_v_i[(n + 3) * n_t_noise + j]
                d_f = input_v_i[(n + 4) * n_t_noise + j]
                n += 5
            end
            if settings["core_source"]
                if settings["core_turbine_attenuation_method"] == "ge"
                    mdoti_c = input_v_i[(n + 0) * n_t_noise + j]
                    Tti_c = input_v_i[(n + 1) * n_t_noise + j]
                    Ttj_c = input_v_i[(n + 2) * n_t_noise + j]
                    Pti_c = input_v_i[(n + 3) * n_t_noise + j]
                    DTt_des_c = input_v_i[(n + 4) * n_t_noise + j]
                    n += 5
                elseif settings["core_turbine_attenuation_method"] == "pw"
                    mdoti_c = input_v_i[(n + 0) * n_t_noise + j]
                    Tti_c = input_v_i[(n + 1) * n_t_noise + j]
                    Ttj_c = input_v_i[(n + 2) * n_t_noise + j]
                    Pti_c = input_v_i[(n + 3) * n_t_noise + j]
                    rho_te_c = input_v_i[(n + 4) * n_t_noise + j]
                    c_te_c = input_v_i[(n + 5) * n_t_noise + j]
                    rho_ti_c = input_v_i[(n + 6) * n_t_noise + j]
                    c_ti_c = input_v_i[(n + 7) * n_t_noise + j]
                    n += 8
                end
            end

            if settings["jet_mixing_source"] == true && settings["jet_shock_source"] == false
                V_j = input_v_i[(n + 0) * n_t_noise + j]
                rho_j = input_v_i[(n + 1) * n_t_noise + j]
                A_j = input_v_i[(n + 2) * n_t_noise + j]
                Tt_j = input_v_i[(n + 3) * n_t_noise + j]
                n += 4
            elseif settings["jet_shock_source"] == true && settings["jet_mixing_source"] == false
                V_j = input_v_i[(n + 0) * n_t_noise + j]
                A_j = input_v_i[(n + 1) * n_t_noise + j]
                Tt_j = input_v_i[(n + 2) * n_t_noise + j]
                M_j = input_v_i[(n + 3) * n_t_noise + j]
                n += 4
            elseif settings["jet_shock_source"] ==true && settings["jet_mixing_source"] == true
                V_j = input_v_i[(n + 0) * n_t_noise + j]
                rho_j = input_v_i[(n + 1) * n_t_noise + j]
                A_j = input_v_i[(n + 2) * n_t_noise + j]
                Tt_j = input_v_i[(n + 3) * n_t_noise + j]
                M_j = input_v_i[(n + 4) * n_t_noise + j]
                n += 5
            end
            if settings["airframe_source"]==true
                theta_flaps = input_v_i[(n + 0) * n_t_noise + j]
                I_landing_gear = input_v_i[(n + 1) * n_t_noise + j]
                n += 2
            end

            # Compute geometry: [r, beta, theta, phi, c_bar, t_o]
            geometry!(geom_v, vcat(x, y, z, alpha, gamma, t_s, c_0, T_0), settings["x_observer_array"][i,:])
            t_o[i,j] = geom_v[6]

            # Compute shielding
            if settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"] && settings["shielding"] == true
                shield .= shielding(settings, data, j, i)
            end
            
            # Compute source
            if settings["fan_inlet_source"]
                fan_source!(spl_j, vcat(DTt_f, mdot_f, N_f, A_f, d_f, c_0, rho_0, M_0, geom_v[3]), settings, pyna_ip, af, f, shield, "fan_inlet")
            end
            if settings["fan_discharge_source"]
                fan_source!(spl_j, vcat(DTt_f, mdot_f, N_f, A_f, d_f, c_0, rho_0, M_0, geom_v[3]), settings, pyna_ip, af, f, shield, "fan_discharge")
            end
            if settings["core_source"]
                if settings["core_turbine_attenuation_method"] == "ge"
                    if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
                        core_source_ge!(spl_j, vcat(mdoti_c, Tti_c, Ttj_c, Pti_c, DTt_des_c, c_0, rho_0, T_0, p_0, M_0, TS, geom_v[3]), settings, pyna_ip, af, f)
                    else
                        core_source_ge!(spl_j, vcat(mdoti_c, Tti_c, Ttj_c, Pti_c, DTt_des_c, c_0, rho_0, T_0, p_0, M_0, 1., geom_v[3]), settings, pyna_ip, af, f)
                    end
                elseif settings["core_turbine_attenuation_method"] == "pw"
                    if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
                        core_source_pw!(spl_j, vcat(mdoti_c, Tti_c, Ttj_c, Pti_c, rho_te_c, c_te_c, rho_ti_c, c_ti_c, c_0, T_0, rho_0, p_0, M_0, TS, geom_v[3]), settings, pyna_ip, af, f)
                    else
                        core_source_pw!(spl_j, vcat(mdoti_c, Tti_c, Ttj_c, Pti_c, rho_te_c, c_te_c, rho_ti_c, c_ti_c, c_0, T_0, rho_0, p_0, M_0, 1., geom_v[3]), settings, pyna_ip, af, f)
                    end
                end
            end
            if settings["jet_mixing_source"]
                if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
                    jet_mixing_source!(spl_j, vcat(V_j, rho_j, A_j, Tt_j, c_0, T_0, rho_0, M_0, TS, geom_v[3]), settings, pyna_ip, af, f)
                else
                    jet_mixing_source!(spl_j, vcat(V_j, rho_j, A_j, Tt_j, c_0, T_0, rho_0, M_0, 1., geom_v[3]), settings, pyna_ip, af, f)
                end
            end
            if settings["jet_shock_source"]
                # Only shock noise if jet Mach number is larger than 1. Choose 1.01 to avoid ill-defined derivatives.
                if M_j > 1.01
                    if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
                        jet_shock_source!(spl_j, vcat(V_j, M_j, A_j, Tt_j, c_0, T_0, M_0, TS, geom_v[3]), settings, pyna_ip, af, f)
                    else
                        jet_shock_source!(spl_j, vcat(V_j, M_j, A_j, Tt_j, c_0, T_0, M_0, 1., geom_v[3]), settings, pyna_ip, af, f)
                    end
                end
            end
            if settings["airframe_source"]
                if M_0 > 0
                    airframe_source!(spl_j, vcat(theta_flaps, I_landing_gear, c_0, rho_0, mu_0, M_0, geom_v[3], geom_v[4]), settings, pyna_ip, af, f)
                end
            end

            # Compute noise propagation
            propagation!(spl_j, vcat(geom_v[1], z, geom_v[5], rho_0, I_0, geom_v[2]), settings, pyna_ip, f_sb, settings["x_observer_array"][i,:])

            # Add spl ambient correction
            f_spl!(spl_j, vcat(c_0, rho_0))
            spl[i,j,:] = spl_j
            
            # Compute noise levels
            if settings["levels_int_metric"] == "ioaspl"
                level[i,j] = f_oaspl(spl_j)
            elseif settings["levels_int_metric"] in ["pnlt_max", "ipnlt", "epnl"]
                level[i,j] = f_pnlt(spl_j, settings, pyna_ip, f)                
            elseif settings["levels_int_metric"] == "sel"
                level[i,j] = f_aspl(spl_j, pyna_ip, f)
            end

        end

        # Compute integrated levels
        if settings["levels_int_metric"] in ["ioaspl", "ipnlt", "sel"]
            level_int[i] = f_ilevel(vcat(t_o[i,:], level[i,:]), settings)
        elseif settings["levels_int_metric"] == "epnl"
            level_int[i] = f_epnl(vcat(t_o[i,:], level[i,:]), settings)
        elseif settings["levels_int_metric"] == "pnlt_max"
            level_int[i] = smooth_max(level[i,:], 50.)
        end

        # Lateral attenuation post-hoc subtraction on integrated noise levels
        if settings["lateral_attenuation"]
            x = input_v_i[0 * n_t_noise + 1: 1 * n_t_noise]
            y = input_v_i[1 * n_t_noise + 1: 2 * n_t_noise]
            z = input_v_i[2 * n_t_noise + 1: 3 * n_t_noise]
            level_int[i] += lateral_attenuation(vcat(x, y, z), settings, settings["x_observer_array"][i,:])
        end

    end

    return t_o, spl, level, level_int

end