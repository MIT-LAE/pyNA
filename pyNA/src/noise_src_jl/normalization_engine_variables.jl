function normalization_engine_variables(settings, n_t, input_norm, comp)
    
    # Extract inputs
    if comp == "jet_mixing"
        V_j = input_norm[0*n_t + 1 : 1*n_t]
        rho_j = input_norm[1*n_t + 1 : 2*n_t]
        A_j = input_norm[2*n_t + 1 : 3*n_t]
        Tt_j = input_norm[3*n_t + 1 : 4*n_t]
        c_0 = input_norm[4*n_t + 1 : 5*n_t]
        rho_0 = input_norm[5*n_t + 1 : 6*n_t]
        T_0 = input_norm[6*n_t + 1 : 7*n_t]
        output_norm = V_j ./ c_0                              # V_j_star
        output_norm = vcat(output_norm, rho_j ./ rho_0)       # rho_j_star
        output_norm = vcat(output_norm, A_j ./ settings.A_e)  # A_j_star
        output_norm = vcat(output_norm, Tt_j ./ T_0)          # Tt_j_star
        
    elseif comp == "jet_shock"
        V_j = input_norm[0*n_t + 1 : 1*n_t]
        A_j = input_norm[1*n_t + 1 : 2*n_t]
        Tt_j = input_norm[2*n_t + 1 : 3*n_t]
        c_0 = input_norm[3*n_t + 1 : 4*n_t]
        T_0 = input_norm[4*n_t + 1 : 5*n_t]
        output_norm = V_j ./ c_0                              # V_j_star
        output_norm = vcat(output_norm, A_j ./ settings.A_e)  # A_j_star
        output_norm = vcat(output_norm, Tt_j ./ T_0)          # Tt_j_star
        
    elseif comp == "jet"
        V_j = input_norm[0*n_t + 1 : 1*n_t]
        rho_j = input_norm[1*n_t + 1 : 2*n_t]
        A_j = input_norm[2*n_t + 1 : 3*n_t]
        Tt_j = input_norm[3*n_t + 1 : 4*n_t]
        c_0 = input_norm[4*n_t + 1 : 5*n_t]
        rho_0 = input_norm[5*n_t + 1 : 6*n_t]
        T_0 = input_norm[6*n_t + 1 : 7*n_t]
        output_norm = V_j ./ c_0                              # V_j_star
        output_norm = vcat(output_norm, rho_j ./ rho_0)       # rho_j_star
        output_norm = vcat(output_norm, A_j ./ settings.A_e)  # A_j_star
        output_norm = vcat(output_norm, Tt_j ./ T_0)          # Tt_j_star
    end
    if comp == "core_ge"
            mdoti_c = input_norm[0*n_t + 1 : 1*n_t]
            Tti_c = input_norm[1*n_t + 1 : 2*n_t]
            Ttj_c = input_norm[2*n_t + 1 : 3*n_t]
            Pti_c = input_norm[3*n_t + 1 : 4*n_t]
            DTt_des_c = input_norm[4*n_t + 1 : 5*n_t]
            c_0 = input_norm[5*n_t + 1 : 6*n_t]
            rho_0 = input_norm[6*n_t + 1 : 7*n_t]
            T_0 = input_norm[7*n_t + 1 : 8*n_t]
            p_0 = input_norm[8*n_t + 1 : 9*n_t]
            output_norm = mdoti_c ./ (rho_0 .* c_0 .* settings.A_e)  # mdoti_c_star
            output_norm = vcat(output_norm, Tti_c ./ T_0)          # Tti_c_star
            output_norm = vcat(output_norm, Ttj_c ./ T_0)          # Ttj_c_star
            output_norm = vcat(output_norm, Pti_c ./ p_0)          # Pti_c_star
            output_norm = vcat(output_norm, DTt_des_c ./ T_0)      # DTt_des_c_star

    elseif comp == "core_pw"
        mdoti_c = input_norm[0*n_t + 1 : 1*n_t]
        Tti_c = input_norm[1*n_t + 1 : 2*n_t]
        Ttj_c = input_norm[2*n_t + 1 : 3*n_t]
        Pti_c = input_norm[3*n_t + 1 : 4*n_t]
        rho_te_c = input_norm[4*n_t + 1 : 5*n_t]
        c_te_c = input_norm[5*n_t + 1 : 6*n_t]
        rho_ti_c = input_norm[6*n_t + 1 : 7*n_t]
        c_ti_c = input_norm[7*n_t + 1 : 8*n_t]
        c_0 = input_norm[8*n_t + 1 : 9*n_t]
        rho_0 = input_norm[9*n_t + 1 : 10*n_t]
        T_0 = input_norm[10*n_t + 1 : 11*n_t]
        p_0 = input_norm[11*n_t + 1 : 12*n_t]
        output_norm = mdoti_c / (c_0 * settings.A_e * rho_0)   # mdoti_c_star
        output_norm = vcat(output_norm, Tti_c ./ T_0)          # Tti_c_star
        output_norm = vcat(output_norm, Ttj_c ./ T_0)          # Ttj_c_star
        output_norm = vcat(output_norm, Pti_c ./ p_0)          # Pti_c_star
        output_norm = vcat(output_norm, rho_te_c ./ rho_0)     # rho_te_c_star
        output_norm = vcat(output_norm, c_te_c ./ c_0)         # c_te_c_star
        output_norm = vcat(output_norm, rho_ti_c ./ rho_0)     # rho_ti_c_star
        output_norm = vcat(output_norm, c_ti_c ./ c_0)         # c_ti_c_star
    end
    if comp == "fan"
        DTt_f = input_norm[0*n_t + 1 : 1*n_t]
        mdot_f = input_norm[1*n_t + 1 : 2*n_t]
        N_f = input_norm[2*n_t + 1 : 3*n_t]
        A_f = input_norm[3*n_t + 1 : 4*n_t]
        d_f = input_norm[4*n_t + 1 : 5*n_t]
        c_0 = input_norm[5*n_t + 1 : 6*n_t]
        rho_0 = input_norm[6*n_t + 1 : 7*n_t]
        T_0 = input_norm[7*n_t + 1 : 8*n_t]
        output_norm = DTt_f ./ T_0                                                # DTt_f_star
        output_norm = vcat(output_norm, mdot_f ./ (rho_0 .* c_0 .* settings.A_e)) # mdot_f_star
        output_norm = vcat(output_norm, N_f ./ (c_0 ./ d_f * 60))                 # N_f_star
        output_norm = vcat(output_norm, A_f ./ settings.A_e)                      # A_f_star
        output_norm = vcat(output_norm, d_f ./ sqrt(settings.A_e))                # d_f_star
        

    end

    return output_norm

end