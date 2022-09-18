using Interpolations: LinearInterpolation, Line
using ReverseDiff


function get_interpolated_input_v!(input_v_i::Array, input_v::Union{Array, ReverseDiff.TrackedArray}, settings, n_t::Int64, n_t_noise::Int64)

    # Extract time vector
    t_s = input_v[5 * n_t + 1 : 6 * n_t]

    # Create interpolated time vector
    dt = (t_s[end]-t_s[1])/(n_t_noise-1)
    t_s_i = zeros(eltype(input_v), n_t_noise)
    for i in range(1, n_t_noise, step=1)
        t_s_i[i] = t_s_i[1] + (i-1)*dt
    end

    # input_v_i = zeros(eltype(input_v), Int64(size(input_v)[1]/n_t*n_t_noise))

    # Interpolate inputs
    f_x = LinearInterpolation(t_s, input_v[0 * n_t + 1 : 1 * n_t], extrapolation_bc=Line())
    f_y = LinearInterpolation(t_s, input_v[1 * n_t + 1 : 2 * n_t], extrapolation_bc=Line())
    f_z = LinearInterpolation(t_s, input_v[2 * n_t + 1 : 3 * n_t], extrapolation_bc=Line())
    f_alpha = LinearInterpolation(t_s, input_v[3 * n_t + 1 : 4 * n_t], extrapolation_bc=Line())
    f_gamma = LinearInterpolation(t_s, input_v[4 * n_t + 1 : 5 * n_t], extrapolation_bc=Line())
    f_M_0 = LinearInterpolation(t_s, input_v[6 * n_t + 1 : 7 * n_t], extrapolation_bc=Line())
    input_v_i[0 * n_t_noise + 1 : 1 * n_t_noise] .= f_x(t_s_i)
    input_v_i[1 * n_t_noise + 1 : 2 * n_t_noise] .= f_y(t_s_i)
    input_v_i[2 * n_t_noise + 1 : 3 * n_t_noise] .= f_z(t_s_i)
    input_v_i[3 * n_t_noise + 1 : 4 * n_t_noise] .= f_alpha(t_s_i)
    input_v_i[4 * n_t_noise + 1 : 5 * n_t_noise] .= f_gamma(t_s_i)
    input_v_i[5 * n_t_noise + 1 : 6 * n_t_noise] .= t_s_i
    input_v_i[6 * n_t_noise + 1 : 7 * n_t_noise] .= f_M_0(t_s_i)
    n = 7

    if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
        f_TS = LinearInterpolation(t_s, input_v[n * n_t + 1 : (n + 1) * n_t], extrapolation_bc=Line())
        input_v_i[(n + 0) * n_t_noise + 1 : (n + 1) * n_t_noise] = f_TS(t_s_i)
        n += 1
    end
    
    f_c_0 = LinearInterpolation(t_s, input_v[(n + 0) * n_t + 1 : (n + 1) * n_t], extrapolation_bc=Line())
    f_T_0 = LinearInterpolation(t_s, input_v[(n + 1) * n_t + 1 : (n + 2) * n_t], extrapolation_bc=Line())
    f_rho_0 = LinearInterpolation(t_s, input_v[(n + 2) * n_t + 1 : (n + 3) * n_t], extrapolation_bc=Line())
    f_p_0 = LinearInterpolation(t_s, input_v[(n + 3) * n_t + 1 : (n + 4) * n_t], extrapolation_bc=Line())
    f_mu_0 = LinearInterpolation(t_s, input_v[(n + 4) * n_t + 1 : (n + 5) * n_t], extrapolation_bc=Line())
    f_I_0 = LinearInterpolation(t_s, input_v[(n + 5) * n_t + 1 : (n + 6) * n_t], extrapolation_bc=Line())
    input_v_i[(n + 0) * n_t_noise + 1 : (n + 1) * n_t_noise] .= f_c_0(t_s_i)
    input_v_i[(n + 1) * n_t_noise + 1 : (n + 2) * n_t_noise] .= f_T_0(t_s_i)
    input_v_i[(n + 2) * n_t_noise + 1 : (n + 3) * n_t_noise] .= f_rho_0(t_s_i)
    input_v_i[(n + 3) * n_t_noise + 1 : (n + 4) * n_t_noise] .= f_p_0(t_s_i)
    input_v_i[(n + 4) * n_t_noise + 1 : (n + 5) * n_t_noise] .= f_mu_0(t_s_i)
    input_v_i[(n + 5) * n_t_noise + 1 : (n + 6) * n_t_noise] .= f_I_0(t_s_i)
    n += 6
    
    if settings["fan_inlet_source"]==true || settings["fan_discharge_source"]==true
        f_DTt_f = LinearInterpolation(t_s, input_v_i[(n + 0) * n_t + 1 : (n + 1) * n_t], extrapolation_bc=Line())
        f_mdot_f = LinearInterpolation(t_s, input_v_i[(n + 1) * n_t + 1 : (n + 2) * n_t], extrapolation_bc=Line())
        f_N_f = LinearInterpolation(t_s, input_v_i[(n + 2) * n_t + 1 : (n + 3) * n_t], extrapolation_bc=Line())
        f_A_f = LinearInterpolation(t_s, input_v_i[(n + 3) * n_t + 1 : (n + 4) * n_t], extrapolation_bc=Line())
        f_d_f = LinearInterpolation(t_s, input_v_i[(n + 4) * n_t + 1 : (n + 5) * n_t], extrapolation_bc=Line())
        input_v_i[(n + 0) * n_t_noise + 1 : (n + 1) * n_t_noise] .= f_DTt_f(t_s_i)
        input_v_i[(n + 1) * n_t_noise + 1 : (n + 2) * n_t_noise] .= f_mdot_f(t_s_i)
        input_v_i[(n + 2) * n_t_noise + 1 : (n + 3) * n_t_noise] .= f_N_f(t_s_i)
        input_v_i[(n + 3) * n_t_noise + 1 : (n + 4) * n_t_noise] .= f_A_f(t_s_i)
        input_v_i[(n + 4) * n_t_noise + 1 : (n + 5) * n_t_noise] .= f_d_f(t_s_i)
        n += 5
    end
    
    if settings["core_source"]
        if settings["core_turbine_attenuation_method"] == "ge"
            f_mdoti_c = LinearInterpolation(t_s, input_v_i[(n + 0) * n_t + 1 : (n + 1) * n_t])
            f_Tti_c = LinearInterpolation(t_s, input_v_i[(n + 1) * n_t + 1 : (n + 2) * n_t])
            f_Ttj_c = LinearInterpolation(t_s, input_v_i[(n + 2) * n_t + 1 : (n + 3) * n_t])
            f_Pti_c = LinearInterpolation(t_s, input_v_i[(n + 3) * n_t + 1 : (n + 4) * n_t])
            f_DTt_des_c = LinearInterpolation(t_s, input_v_i[(n + 4) * n_t + 1 : (n + 5) * n_t])
            input_v_i[(n + 0) * n_t_noise + 1 : (n + 1) * n_t_noise] .= f_mdoti_c(t_s_i)
            input_v_i[(n + 1) * n_t_noise + 1 : (n + 2) * n_t_noise] .= f_Tti_c(t_s_i)
            input_v_i[(n + 2) * n_t_noise + 1 : (n + 3) * n_t_noise] .= f_Ttj_c(t_s_i)
            input_v_i[(n + 3) * n_t_noise + 1 : (n + 4) * n_t_noise] .= f_Pti_c(t_s_i)
            input_v_i[(n + 4) * n_t_noise + 1 : (n + 5) * n_t_noise] .= f_DTt_des_c(t_s_i)
            n += 5
        elseif settings["core_turbine_attenuation_method"] == "pw"
            f_mdoti_c = LinearInterpolation(t_s, input_v_i[(n + 0) * n_t + 1 : (n + 1) * n_t])
            f_Tti_c = LinearInterpolation(t_s, input_v_i[(n + 1) * n_t + 1 : (n + 2) * n_t])
            f_Ttj_c = LinearInterpolation(t_s, input_v_i[(n + 2) * n_t + 1 : (n + 3) * n_t])
            f_Pti_c = LinearInterpolation(t_s, input_v_i[(n + 3) * n_t + 1 : (n + 4) * n_t])
            f_rho_te_c = LinearInterpolation(t_s, input_v_i[(n + 4) * n_t + 1 : (n + 5) * n_t])
            f_c_te_c = LinearInterpolation(t_s, input_v_i[(n + 5) * n_t + 1 : (n + 6) * n_t])
            f_rho_ti_c = LinearInterpolation(t_s, input_v_i[(n + 6) * n_t + 1 : (n + 7) * n_t])
            f_c_ti_c = LinearInterpolation(t_s, input_v_i[(n + 7) * n_t + 1 : (n + 8) * n_t])
            input_v_i[(n + 0) * n_t_noise + 1 : (n + 1) * n_t_noise] .= f_mdoti_c(t_s_i)
            input_v_i[(n + 1) * n_t_noise + 1 : (n + 2) * n_t_noise] .= f_Tti_c(t_s_i)
            input_v_i[(n + 2) * n_t_noise + 1 : (n + 3) * n_t_noise] .= f_Ttj_c(t_s_i)
            input_v_i[(n + 3) * n_t_noise + 1 : (n + 4) * n_t_noise] .= f_Pti_c(t_s_i)
            input_v_i[(n + 4) * n_t_noise + 1 : (n + 5) * n_t_noise] .= f_rho_te_c(t_s_i)
            input_v_i[(n + 5) * n_t_noise + 1 : (n + 6) * n_t_noise] .= f_c_te_c(t_s_i)
            input_v_i[(n + 6) * n_t_noise + 1 : (n + 7) * n_t_noise] .= f_rho_ti_c(t_s_i)
            input_v_i[(n + 7) * n_t_noise + 1 : (n + 8) * n_t_noise] .= f_c_ti_c(t_s_i)
            n += 8
        end
    end

    if settings["jet_mixing_source"] == true && settings["jet_shock_source"] == false
        f_V_j = LinearInterpolation(t_s, input_v[(n + 0) * n_t + 1 : (n + 1) * n_t], extrapolation_bc=Line())
        f_rho_j = LinearInterpolation(t_s, input_v[(n + 1) * n_t + 1 : (n + 2) * n_t], extrapolation_bc=Line())
        f_A_j = LinearInterpolation(t_s, input_v[(n + 2) * n_t + 1 : (n + 3) * n_t], extrapolation_bc=Line())
        f_Tt_j = LinearInterpolation(t_s, input_v[(n + 3) * n_t + 1 : (n + 4) * n_t], extrapolation_bc=Line())
        input_v_i[(n + 0) * n_t_noise + 1 : (n + 1) * n_t_noise] .= f_V_j(t_s_i)
        input_v_i[(n + 1) * n_t_noise + 1 : (n + 2) * n_t_noise] .= f_rho_j(t_s_i)
        input_v_i[(n + 2) * n_t_noise + 1 : (n + 3) * n_t_noise] .= f_A_j(t_s_i)
        input_v_i[(n + 3) * n_t_noise + 1 : (n + 4) * n_t_noise] .= f_Tt_j(t_s_i)
        n += 4
    elseif settings["jet_shock_source"] == true && settings["jet_mixing_source"] == false
        f_V_j = LinearInterpolation(t_s, input_v[(n + 0) * n_t + 1 : (n + 1) * n_t], extrapolation_bc=Line())
        f_A_j = LinearInterpolation(t_s, input_v[(n + 1) * n_t + 1 : (n + 2) * n_t], extrapolation_bc=Line())
        f_Tt_j = LinearInterpolation(t_s, input_v[(n + 2) * n_t + 1 : (n + 3) * n_t], extrapolation_bc=Line())
        f_M_j = LinearInterpolation(t_s, input_v[(n + 3) * n_t + 1 : (n + 4) * n_t], extrapolation_bc=Line())
        input_v_i[(n + 0) * n_t_noise + 1 : (n + 1) * n_t_noise] .= f_V_j(t_s_i)
        input_v_i[(n + 1) * n_t_noise + 1 : (n + 2) * n_t_noise] .= f_A_j(t_s_i)
        input_v_i[(n + 2) * n_t_noise + 1 : (n + 3) * n_t_noise] .= f_Tt_j(t_s_i)
        input_v_i[(n + 3) * n_t_noise + 1 : (n + 4) * n_t_noise] .= f_M_j(t_s_i)
        n += 4
    elseif settings["jet_shock_source"] ==true && settings["jet_mixing_source"] == true
        f_V_j = LinearInterpolation(t_s, input_v[(n + 0) * n_t + 1 : (n + 1) * n_t], extrapolation_bc=Line())
        f_rho_j = LinearInterpolation(t_s, input_v[(n + 1) * n_t + 1 : (n + 2) * n_t], extrapolation_bc=Line())
        f_A_j = LinearInterpolation(t_s, input_v[(n + 2) * n_t + 1 : (n + 3) * n_t], extrapolation_bc=Line())
        f_Tt_j = LinearInterpolation(t_s, input_v[(n + 3) * n_t + 1 : (n + 4) * n_t], extrapolation_bc=Line())
        f_M_j = LinearInterpolation(t_s, input_v[(n + 4) * n_t + 1 : (n + 5) * n_t], extrapolation_bc=Line())
        input_v_i[(n + 0) * n_t_noise + 1 : (n + 1) * n_t_noise] .= f_V_j(t_s_i)
        input_v_i[(n + 1) * n_t_noise + 1 : (n + 2) * n_t_noise] .= f_rho_j(t_s_i)
        input_v_i[(n + 2) * n_t_noise + 1 : (n + 3) * n_t_noise] .= f_A_j(t_s_i)
        input_v_i[(n + 3) * n_t_noise + 1 : (n + 4) * n_t_noise] .= f_Tt_j(t_s_i)
        input_v_i[(n + 4) * n_t_noise + 1 : (n + 5) * n_t_noise] .= f_M_j(t_s_i)
        n += 5
    end

    if settings["airframe_source"]==true
        f_theta_flaps = LinearInterpolation(t_s, input_v[n * n_t + 1 : (n + 1) * n_t], extrapolation_bc=Line())
        f_I_landing_gear = LinearInterpolation(t_s, input_v[(n + 1) * n_t + 1 : (n + 2) * n_t], extrapolation_bc=Line())
        input_v_i[(n + 0) * n_t_noise + 1 : (n + 1) * n_t_noise] .= f_theta_flaps(t_s_i)
        input_v_i[(n + 1) * n_t_noise + 1 : (n + 2) * n_t_noise] .= f_I_landing_gear(t_s_i)
        n += 2
    end

end

get_interpolated_input_v_fwd! = (y,x) -> get_interpolated_input_v!(y, x, settings, n_t, n_t_noise)