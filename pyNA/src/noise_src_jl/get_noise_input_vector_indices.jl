function get_noise_input_vector_indices(settings, n_t)
    
    """
    Get indices for input vector of noise model.

    :param settings: noise settings
    :type settings: dict
    :param n_t: number of time steps in trajectory
    :type n_t: int

    :return: idx

    """

    # Initialize indices dictionary
    idx = Dict()

    idx["x"] = [0 * n_t + 1, 1 * n_t]
    idx["y"] = [1 * n_t + 1, 2 * n_t]
    idx["z"] = [2 * n_t + 1, 3 * n_t]
    idx["alpha"] = [3 * n_t + 1, 4 * n_t]
    idx["gamma"] = [4 * n_t + 1, 5 * n_t]
    idx["t_s"] = [5 * n_t + 1, 6 * n_t]
    idx["M_0"] = [6 * n_t + 1, 7 * n_t]
    n = 7

    if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
        idx["TS"] = [n * n_t + 1, (n + 1) * n_t]
        n = n + 1
    end
    
    if settings["atmosphere_type"] == "stratified"
        idx["c_0"]   = [n * n_t + 1, (n + 1) * n_t]
        idx["T_0"]   = [(n + 1) * n_t + 1, (n + 2) * n_t]
        idx["rho_0"] = [(n + 2) * n_t + 1, (n + 3) * n_t]
        idx["p_0"]   = [(n + 3) * n_t + 1, (n + 4) * n_t]
        idx["mu_0"]  = [(n + 4) * n_t + 1, (n + 5) * n_t]
        idx["I_0"]   = [(n + 5) * n_t + 1, (n + 6) * n_t]
        n = n + 6
    end

    if settings["fan_inlet_source"] == true || settings["fan_discharge_source"] == true
        idx["DTt_f"]  = [n * n_t + 1, (n + 1) * n_t]
        idx["mdot_f"] = [(n + 1) * n_t + 1, (n + 2) * n_t]
        idx["N_f"]    = [(n + 2) * n_t + 1, (n + 3) * n_t]
        idx["A_f"]    = [(n + 3) * n_t + 1, (n + 4) * n_t]
        idx["d_f"]    = [(n + 4) * n_t + 1, (n + 5) * n_t]
        n = n + 5
    end

    if settings["core_source"]
        if settings["core_turbine_attenuation_method"] == "ge"
            idx["mdoti_c"]   = [n * n_t + 1, (n + 1) * n_t]
            idx["Tti_c"]     = [(n + 1) * n_t + 1, (n + 2) * n_t]
            idx["Ttj_c"]     = [(n + 2) * n_t + 1, (n + 3) * n_t]
            idx["Pti_c"]     = [(n + 3) * n_t + 1, (n + 4) * n_t]
            idx["DTt_des_c"] = [(n + 4) * n_t + 1, (n + 5) * n_t]
            n = n + 5
        elseif settings["core_turbine_attenuation_method"] == "pw"
            idx["mdoti_c"]  = [n * n_t + 1, (n + 1) * n_t]
            idx["Tti_c"]    = [(n + 1) * n_t + 1, (n + 2) * n_t]
            idx["Ttj_c"]    = [(n + 2) * n_t + 1, (n + 3) * n_t]
            idx["Pti_c"]    = [(n + 3) * n_t + 1, (n + 4) * n_t]
            idx["rho_te_c"] = [(n + 4) * n_t + 1, (n + 5) * n_t]
            idx["c_te_c"]   = [(n + 5) * n_t + 1, (n + 6) * n_t]
            idx["rho_ti_c"] = [(n + 6) * n_t + 1, (n + 7) * n_t]
            idx["c_ti_c"]   = [(n + 7) * n_t + 1, (n + 8) * n_t]
            n = n + 8
        end
    end

    if settings["jet_mixing_source"] == true && settings["jet_shock_source"] == false
        idx["V_j"]   = [n * n_t + 1, (n + 1) * n_t]
        idx["rho_j"] = [(n + 1) * n_t + 1, (n + 2) * n_t]
        idx["A_j"]   = [(n + 2) * n_t + 1, (n + 3) * n_t]
        idx["Tt_j"]  = [(n + 3) * n_t + 1, (n + 4) * n_t]
        n = n + 4
    elseif settings["jet_shock_source"] == true && settings["jet_mixing_source"] == false
        idx["V_j"]  = [n * n_t + 1, (n + 1) * n_t]
        idx["M_j"]  = [(n + 1) * n_t + 1, (n + 2) * n_t]
        idx["A_j"]  = [(n + 2) * n_t + 1, (n + 3) * n_t]
        idx["Tt_j"] = [(n + 3) * n_t + 1, (n + 4) * n_t]
        n = n + 4
    elseif settings["jet_shock_source"] == true && settings["jet_mixing_source"] == true
        idx["V_j"]   = [n * n_t + 1, (n + 1) * n_t]
        idx["rho_j"] = [(n + 1) * n_t + 1, (n + 2) * n_t]
        idx["A_j"]   = [(n + 2) * n_t + 1, (n + 3) * n_t]
        idx["Tt_j"]  = [(n + 3) * n_t + 1, (n + 4) * n_t]
        idx["M_j"]   = [(n + 4) * n_t + 1, (n + 5) * n_t]
        n = n + 5
    end

    if settings["airframe_source"]
        idx["theta_flaps"]    = [n * n_t + 1, (n + 1) * n_t]
        idx["I_landing_gear"] = [(n + 1) * n_t + 1, (n + 2) * n_t]
        n = n + 2
    end

    return idx

end