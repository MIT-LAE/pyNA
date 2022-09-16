using OpenMDAO: AbstractExplicitComp, VarData, PartialsData, get_rows_cols
import OpenMDAO: setup, compute!, compute_partials!
using Interpolations
using ReverseDiff
using LinearAlgebra
using ConcreteStructs
using PCHIPInterpolation
using Dates
using FLoops
using BenchmarkTools

include("get_interpolation_functions.jl")
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


function get_noise(input_v, settings, pyna_ip, af, data, sealevel_atmosphere, idx_input_v)
    
    # Get type of input vector
    T = eltype(input_v)

    # 1/3rd octave band frequencies
    f = data.f
    f_sb = data.f_sb

    # Number of time steps
    n_t = size(input_v[idx_input_v["t_s"][1]:idx_input_v["t_s"][2]])[1]

    # Number of observers
    n_obs = size(settings["x_observer_array"])[1]
    
    # # Initialize outputs
    t_o = zeros(T, (n_obs, n_t))
    spl = 1e-99*ones(T, (n_obs, n_t, settings["n_frequency_bands"]))
    level = zeros(T, (n_obs, n_t))
    level_int = zeros(eltype(input_v), n_obs)

    # Iterate over observers
    for i in 1:1:n_obs
        
        println("Computing noise at observer: ", i)

        # Iterate over time steps
        for j in 1:1:n_t

            # Extract inputs
            x = input_v[idx_input_v["x"][1]:idx_input_v["x"][2]][j]
            y = input_v[idx_input_v["y"][1]:idx_input_v["y"][2]][j]
            z = input_v[idx_input_v["z"][1]:idx_input_v["z"][2]][j]
            alpha = input_v[idx_input_v["alpha"][1]:idx_input_v["alpha"][2]][j]
            gamma = input_v[idx_input_v["gamma"][1]:idx_input_v["gamma"][2]][j]
            t_s = input_v[idx_input_v["t_s"][1]:idx_input_v["t_s"][2]][j]
            M_0 = input_v[idx_input_v["M_0"][1]:idx_input_v["M_0"][2]][j]
            if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
                TS = input_v[idx_input_v["TS"][1]:idx_input_v["TS"][2]][j]
            end
            if settings["atmosphere_type"] == "stratified"
                c_0 = input_v[idx_input_v["c_0"][1]:idx_input_v["c_0"][2]][j]
                T_0 = input_v[idx_input_v["T_0"][1]:idx_input_v["T_0"][2]][j]
                rho_0 = input_v[idx_input_v["rho_0"][1]:idx_input_v["rho_0"][2]][j]
                p_0 = input_v[idx_input_v["p_0"][1]:idx_input_v["p_0"][2]][j]
                mu_0 = input_v[idx_input_v["mu_0"][1]:idx_input_v["mu_0"][2]][j]
                I_0 = input_v[idx_input_v["I_0"][1]:idx_input_v["I_0"][2]][j]
            else
                c_0 = sealevel_atmosphere["c_0"]
                T_0 = sealevel_atmosphere["T_0"]
                rho_0 = sealevel_atmosphere["rho_0"]
                p_0 = sealevel_atmosphere["p_0"]
                mu_0 = sealevel_atmosphere["mu_0"]
                I_0 = sealevel_atmosphere["I_0"]
            end  
            if settings["jet_mixing_source"] == true && settings["jet_shock_source"] == false
                V_j = input_v[idx_input_v["V_j"][1]:idx_input_v["V_j"][2]][j]
                rho_j = input_v[idx_input_v["rho_j"][1]:idx_input_v["rho_j"][2]][j]
                A_j = input_v[idx_input_v["A_j"][1]:idx_input_v["A_j"][2]][j]
                Tt_j = input_v[idx_input_v["Tt_j"][1]:idx_input_v["Tt_j"][2]][j]
            elseif settings["jet_shock_source"] == true && settings["jet_mixing_source"] == false
                V_j = input_v[idx_input_v["V_j"][1]:idx_input_v["V_j"][2]][j]
                A_j = input_v[idx_input_v["A_j"][1]:idx_input_v["A_j"][2]][j]
                Tt_j = input_v[idx_input_v["Tt_j"][1]:idx_input_v["Tt_j"][2]][j]
                M_j = input_v[idx_input_v["M_j"][1]:idx_input_v["M_j"][2]][j]
            elseif settings["jet_shock_source"] ==true && settings["jet_mixing_source"] == true
                V_j = input_v[idx_input_v["V_j"][1]:idx_input_v["V_j"][2]][j]
                rho_j = input_v[idx_input_v["rho_j"][1]:idx_input_v["rho_j"][2]][j]
                A_j = input_v[idx_input_v["A_j"][1]:idx_input_v["A_j"][2]][j]
                Tt_j = input_v[idx_input_v["Tt_j"][1]:idx_input_v["Tt_j"][2]][j]
                M_j = input_v[idx_input_v["M_j"][1]:idx_input_v["M_j"][2]][j]
            end
            if settings["core_source"]
                if settings["core_turbine_attenuation_method"] == "ge"
                    mdoti_c = input_v[idx_input_v["mdoti_c"][1]:idx_input_v["mdoti_c"][2]][j]
                    Tti_c = input_v[idx_input_v["Tti_c"][1]:idx_input_v["Tti_c"][2]][j]
                    Ttj_c = input_v[idx_input_v["Ttj_c"][1]:idx_input_v["Ttj_c"][2]][j]
                    Pti_c = input_v[idx_input_v["Pti_c"][1]:idx_input_v["Pti_c"][2]][j]
                    DTt_des_c = input_v[idx_input_v["DTt_des_c"][1]:idx_input_v["DTt_des_c"][2]][j]
                elseif settings["core_turbine_attenuation_method"] == "pw"
                    mdoti_c = input_v[idx_input_v["mdoti_c"][1]:idx_input_v["mdoti_c"][2]][j]
                    Tti_c = input_v[idx_input_v["Tti_c"][1]:idx_input_v["Tti_c"][2]][j]
                    Ttj_c = input_v[idx_input_v["Ttj_c"][1]:idx_input_v["Ttj_c"][2]][j]
                    Pti_c = input_v[idx_input_v["Pti_c"][1]:idx_input_v["Pti_c"][2]][j]
                    rho_te_c_star = input_v[idx_input_v["rho_te_c"][1]:idx_input_v["rho_te_c"][2]][j]
                    c_te_c_star = input_v[idx_input_v["c_te_c"][1]:idx_input_v["c_te_c"][2]][j]
                    rho_ti_c_star = input_v[idx_input_v["rho_ti_c"][1]:idx_input_v["rho_ti_c"][2]][j]
                    c_ti_c_star = input_v[idx_input_v["c_ti_c"][1]:idx_input_v["c_ti_c"][2]][j]
                end
            end
            if settings["fan_inlet_source"]==true || settings["fan_discharge_source"]==true
                DTt_f = input_v[idx_input_v["DTt_f"][1]:idx_input_v["DTt_f"][2]][j]
                mdot_f = input_v[idx_input_v["mdot_f"][1]:idx_input_v["mdot_f"][2]][j]
                N_f = input_v[idx_input_v["N_f"][1]:idx_input_v["N_f"][2]][j]
                A_f = input_v[idx_input_v["A_f"][1]:idx_input_v["A_f"][2]][j]
                d_f = input_v[idx_input_v["d_f"][1]:idx_input_v["d_f"][2]][j]
            end
            if settings["airframe_source"]==true
                theta_flaps = input_v[idx_input_v["theta_flaps"][1]:idx_input_v["theta_flaps"][2]][j]
                I_landing_gear = input_v[idx_input_v["I_landing_gear"][1]:idx_input_v["I_landing_gear"][2]][j]
            end

            # Compute geometry: [r, beta, theta, phi, c_bar, t_o]
            geom_v = zeros(eltype(input_v), 6)
            geometry!(geom_v, vcat(x, y, z, alpha, gamma, t_s, c_0, T_0), settings["x_observer_array"][i,:])
            t_o[i,j] = geom_v[6]

            # shielding
            shield = shielding(settings, data, j, i)

            # Compute source
            spl_j = zeros(eltype(input_v), settings["n_frequency_bands"])
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
            x = input_v[idx_input_v["x"][1]:idx_input_v["x"][2]]
            y = input_v[idx_input_v["y"][1]:idx_input_v["y"][2]]
            z = input_v[idx_input_v["z"][1]:idx_input_v["z"][2]]
            level_int[i] += lateral_attenuation(vcat(x, y, z), settings, settings["x_observer_array"][i,:])
        end

    end

    # Write to output_v
    return t_o, spl, level, level_int

end

function get_noise2!(output_v, input_v, settings, pyna_ip, af, data, sealevel_atmosphere, idx_input_v)
    
    # Get type of input vector
    T = eltype(input_v)

    # 1/3rd octave band frequencies
    f = data.f
    f_sb = data.f_sb

    # Number of time steps
    n_t = size(input_v[idx_input_v["t_s"][1]:idx_input_v["t_s"][2]])[1]

    # Number of observers
    n_obs = size(settings["x_observer_array"])[1]
    
    # # Initialize outputs
    t_o = zeros(T, (n_obs, n_t))
    level = zeros(T, (n_obs, n_t))
    level_int = zeros(eltype(input_v), n_obs)

    # Iterate over observers
    for i in 1:1:n_obs
        
        println("Computing noise at observer: ", i)

        # Iterate over time steps
        for j in 1:1:n_t

            # Extract inputs
            x = input_v[idx_input_v["x"][1]:idx_input_v["x"][2]][j]
            y = input_v[idx_input_v["y"][1]:idx_input_v["y"][2]][j]
            z = input_v[idx_input_v["z"][1]:idx_input_v["z"][2]][j]
            alpha = input_v[idx_input_v["alpha"][1]:idx_input_v["alpha"][2]][j]
            gamma = input_v[idx_input_v["gamma"][1]:idx_input_v["gamma"][2]][j]
            t_s = input_v[idx_input_v["t_s"][1]:idx_input_v["t_s"][2]][j]
            M_0 = input_v[idx_input_v["M_0"][1]:idx_input_v["M_0"][2]][j]
            if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
                TS = input_v[idx_input_v["TS"][1]:idx_input_v["TS"][2]][j]
            end
            if settings["atmosphere_type"] == "stratified"
                c_0 = input_v[idx_input_v["c_0"][1]:idx_input_v["c_0"][2]][j]
                T_0 = input_v[idx_input_v["T_0"][1]:idx_input_v["T_0"][2]][j]
                rho_0 = input_v[idx_input_v["rho_0"][1]:idx_input_v["rho_0"][2]][j]
                p_0 = input_v[idx_input_v["p_0"][1]:idx_input_v["p_0"][2]][j]
                mu_0 = input_v[idx_input_v["mu_0"][1]:idx_input_v["mu_0"][2]][j]
                I_0 = input_v[idx_input_v["I_0"][1]:idx_input_v["I_0"][2]][j]
            else
                c_0 = sealevel_atmosphere["c_0"]
                T_0 = sealevel_atmosphere["T_0"]
                rho_0 = sealevel_atmosphere["rho_0"]
                p_0 = sealevel_atmosphere["p_0"]
                mu_0 = sealevel_atmosphere["mu_0"]
                I_0 = sealevel_atmosphere["I_0"]
            end  
            if settings["jet_mixing_source"] == true && settings["jet_shock_source"] == false
                V_j = input_v[idx_input_v["V_j"][1]:idx_input_v["V_j"][2]][j]
                rho_j = input_v[idx_input_v["rho_j"][1]:idx_input_v["rho_j"][2]][j]
                A_j = input_v[idx_input_v["A_j"][1]:idx_input_v["A_j"][2]][j]
                Tt_j = input_v[idx_input_v["Tt_j"][1]:idx_input_v["Tt_j"][2]][j]
            elseif settings["jet_shock_source"] == true && settings["jet_mixing_source"] == false
                V_j = input_v[idx_input_v["V_j"][1]:idx_input_v["V_j"][2]][j]
                A_j = input_v[idx_input_v["A_j"][1]:idx_input_v["A_j"][2]][j]
                Tt_j = input_v[idx_input_v["Tt_j"][1]:idx_input_v["Tt_j"][2]][j]
                M_j = input_v[idx_input_v["M_j"][1]:idx_input_v["M_j"][2]][j]
            elseif settings["jet_shock_source"] ==true && settings["jet_mixing_source"] == true
                V_j = input_v[idx_input_v["V_j"][1]:idx_input_v["V_j"][2]][j]
                rho_j = input_v[idx_input_v["rho_j"][1]:idx_input_v["rho_j"][2]][j]
                A_j = input_v[idx_input_v["A_j"][1]:idx_input_v["A_j"][2]][j]
                Tt_j = input_v[idx_input_v["Tt_j"][1]:idx_input_v["Tt_j"][2]][j]
                M_j = input_v[idx_input_v["M_j"][1]:idx_input_v["M_j"][2]][j]
            end
            if settings["core_source"]
                if settings["core_turbine_attenuation_method"] == "ge"
                    mdoti_c = input_v[idx_input_v["mdoti_c"][1]:idx_input_v["mdoti_c"][2]][j]
                    Tti_c = input_v[idx_input_v["Tti_c"][1]:idx_input_v["Tti_c"][2]][j]
                    Ttj_c = input_v[idx_input_v["Ttj_c"][1]:idx_input_v["Ttj_c"][2]][j]
                    Pti_c = input_v[idx_input_v["Pti_c"][1]:idx_input_v["Pti_c"][2]][j]
                    DTt_des_c = input_v[idx_input_v["DTt_des_c"][1]:idx_input_v["DTt_des_c"][2]][j]
                elseif settings["core_turbine_attenuation_method"] == "pw"
                    mdoti_c = input_v[idx_input_v["mdoti_c"][1]:idx_input_v["mdoti_c"][2]][j]
                    Tti_c = input_v[idx_input_v["Tti_c"][1]:idx_input_v["Tti_c"][2]][j]
                    Ttj_c = input_v[idx_input_v["Ttj_c"][1]:idx_input_v["Ttj_c"][2]][j]
                    Pti_c = input_v[idx_input_v["Pti_c"][1]:idx_input_v["Pti_c"][2]][j]
                    rho_te_c_star = input_v[idx_input_v["rho_te_c"][1]:idx_input_v["rho_te_c"][2]][j]
                    c_te_c_star = input_v[idx_input_v["c_te_c"][1]:idx_input_v["c_te_c"][2]][j]
                    rho_ti_c_star = input_v[idx_input_v["rho_ti_c"][1]:idx_input_v["rho_ti_c"][2]][j]
                    c_ti_c_star = input_v[idx_input_v["c_ti_c"][1]:idx_input_v["c_ti_c"][2]][j]
                end
            end
            if settings["fan_inlet_source"]==true || settings["fan_discharge_source"]==true
                DTt_f = input_v[idx_input_v["DTt_f"][1]:idx_input_v["DTt_f"][2]][j]
                mdot_f = input_v[idx_input_v["mdot_f"][1]:idx_input_v["mdot_f"][2]][j]
                N_f = input_v[idx_input_v["N_f"][1]:idx_input_v["N_f"][2]][j]
                A_f = input_v[idx_input_v["A_f"][1]:idx_input_v["A_f"][2]][j]
                d_f = input_v[idx_input_v["d_f"][1]:idx_input_v["d_f"][2]][j]
            end
            if settings["airframe_source"]==true
                theta_flaps = input_v[idx_input_v["theta_flaps"][1]:idx_input_v["theta_flaps"][2]][j]
                I_landing_gear = input_v[idx_input_v["I_landing_gear"][1]:idx_input_v["I_landing_gear"][2]][j]
            end

            # Compute geometry: [r, beta, theta, phi, c_bar, t_o]
            geom_v = zeros(eltype(input_v), 6)
            geometry!(geom_v, vcat(x, y, z, alpha, gamma, t_s, c_0, T_0), settings["x_observer_array"][i,:])
            t_o[i,j] = geom_v[6]

            # shielding
            shield = shielding(settings, data, j, i)

            # Compute source
            spl_j = zeros(eltype(input_v), settings["n_frequency_bands"])
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

            # Compute noise levels
            if settings["levels_int_metric"] == "ioaspl"
                level[i,j] = f_oaspl(spl_j)
        #     elseif settings["levels_int_metric"] in ["pnlt_max", "ipnlt", "epnl"]
        #         level[i,j] = f_pnlt(spl_j, settings, pyna_ip, f)                
        #     elseif settings["levels_int_metric"] == "sel"
        #         level[i,j] = f_aspl(spl_j, pyna_ip, f)
            end

        end

        # # Compute integrated levels
        # if settings["levels_int_metric"] in ["ioaspl", "ipnlt", "sel"]
        #     level_int[i] = f_ilevel(vcat(t_o[i,:], level[i,:]), settings)
        # elseif settings["levels_int_metric"] == "epnl"
        #     level_int[i] = f_epnl(vcat(t_o[i,:], level[i,:]), settings)
        # elseif settings["levels_int_metric"] == "pnlt_max"
        #     level_int[i] = smooth_max(level[i,:], 50.)
        # end

        # # Lateral attenuation post-hoc subtraction on integrated noise levels
        # if settings["lateral_attenuation"]
        #     x = input_v[idx_input_v["x"][1]:idx_input_v["x"][2]]
        #     y = input_v[idx_input_v["y"][1]:idx_input_v["y"][2]]
        #     z = input_v[idx_input_v["z"][1]:idx_input_v["z"][2]]
        #     level_int[i] += lateral_attenuation(vcat(x, y, z), settings, settings["x_observer_array"][i,:])
        # end

    end

    # Compute output_v: [level_int_lateral, level_int_flyover]
    # level_lateral = smooth_max(level_int[1:end-1], 50.)
    # objective = level_int[end]
    # output_v .= [level_lateral, objective]

    println(size(level))

    output_v .= sum(sum(level, dims=1), dims=2)

end