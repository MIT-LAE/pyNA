function normalization_jet_mixing!(settings, V_j, rho_j, A_j, Tt_j, c_0, rho_0, T_0)

    @. V_j = V_j / c_0 
    @. rho_j = rho_j / rho_0
    @. A_j = A_j / settings.A_e
    @. Tt_j = Tt_j / T_0

end

function normalization_jet_shock!(settings, V_j, A_j, Tt_j, c_0, T_0)
    
    @. V_j = V_j / c_0 
    @. A_j = A_j / settings.A_e
    @. Tt_j = Tt_j / T_0

end

function normalization_jet!(settings, V_j, rho_j, A_j, Tt_j, c_0, rho_0, T_0)

    @. V_j = V_j / c_0 
    @. rho_j = rho_j / rho_0
    @. A_j = A_j / settings.A_e
    @. Tt_j = Tt_j / T_0

end

function normalization_core_ge!(settings, mdoti_c, Tti_c, Ttj_c, Pti_c, DTt_des_c, c_0, rho_0, T_0, p_0)

    @. mdoti_c = mdoti_c / (rho_0 * c_0 * settings.A_e)
    @. Tti_c = Tti_c / T_0
    @. Ttj_c = Ttj_c / T_0
    @. Pti_c = Pti_c / p_0
    @. DTt_des_c = DTt_des_c / T_0

end

function normalization_core_pw!(settings, mdoti_c, Tti_c, Ttj_c, Pti_c, rho_te_c, c_te_c, rho_ti_c, c_ti_c, c_0, rho_0, T_0, p_0)

    @. mdoti_c = mdoti_c / (c_0 * settings.A_e * rho_0)
    @. Tti_c = Tti_c / T_0
    @. Ttj_c = Ttj_c / T_0
    @. Pti_c = Pti_c / p_0
    @. rho_te_c = rho_te_c / rho_0
    @. c_te_c = c_te_c / c_0
    @. rho_ti_c = rho_ti_c / rho_0
    @. c_ti_c = c_ti_c / c_0

end

function normalization_fan!(settings, DTt_f ,mdot_f, N_f, A_f, d_f, c_0, rho_0, T_0)

    @. DTt_f = DTt_f / T_0
    @. mdot_f = mdot_f / (rho_0 * c_0 * settings.A_e)
    @. N_f = N_f / (c_0 / d_f * 60)
    @. A_f = A_f / settings.A_e
    @. d_f = d_f / sqrt(settings.A_e)

end
