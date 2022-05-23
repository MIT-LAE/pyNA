function source(settings, data, ac, shielding, n_t, idx_src, input_src)

    # Extract inputs
    TS = input_src[idx_src["TS"][1]:idx_src["TS"][2]]

    # Get type of input vector
    T = eltype(input_src)

    # Initialize source mean-square acoustic pressure
    msap_source = zeros(T, (n_t, settings.N_f))

    if settings.fan_inlet
        msap_fan_inlet = fan(settings, data, ac, n_t, shielding, idx_src, input_src, "fan_inlet")
        msap_source = msap_source .+ msap_fan_inlet
    end
    if settings.fan_discharge
        msap_fan_discharge = fan(settings, data, ac, n_t, shielding, idx_src, input_src, "fan_discharge")
        msap_source = msap_source .+ msap_fan_discharge
    end

    if settings.core
        msap_core = core(settings, data, ac, n_t, idx_src, input_src)
        if settings.suppression && settings.case_name in ["nasa_stca_standard", "stca_enginedesign_standard"]
            msap_core[findall(TS.*ones(1, settings.N_f).>0.8)] = (10. ^(-2.3 / 10.) * msap_core)[findall(TS.*ones(1, settings.N_f).>0.8)]
        end
        msap_source = msap_source .+ msap_core
    end

    if settings.jet_mixing 
        msap_jet_mixing = jet_mixing(settings, data, ac, n_t, idx_src, input_src)
        if settings.suppression && settings.case_name in ["nasa_stca_standard", "stca_enginedesign_standard"]
            msap_jet_mixing[findall(TS.*ones(1, settings.N_f).>0.8)] = (10. ^(-2.3 / 10.) * msap_jet_mixing)[findall(TS.*ones(1, settings.N_f).>0.8)]
        end
        msap_source = msap_source .+ msap_jet_mixing
    end
    if settings.jet_shock
        msap_jet_shock = jet_shock(settings, data, ac, n_t, idx_src, input_src)
        if settings.suppression && settings.case_name in ["nasa_stca_standard", "stca_enginedesign_standard"]
            msap_jet_shock[findall(TS.*ones(1, settings.N_f).>0.8)] = (10. ^(-2.3 / 10.) * msap_jet_shock)[findall(TS.*ones(1, settings.N_f).>0.8)]
        end
        msap_source = msap_source .+ msap_jet_shock
    end

    if settings.airframe
        msap_af = airframe(settings, data, ac, n_t, idx_src, input_src)
        msap_source = msap_source .+ msap_af
    end

    return msap_source
end

