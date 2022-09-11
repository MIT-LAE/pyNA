function atmospheric_absorption!(spl_sb, pyna_ip, settings, f_sb, z, r)
    # Calculate average absorption factor between observer and source
    alpha = pyna_ip.f_abs.(z.*ones(settings["n_frequency_subbands"]*settings["n_frequency_bands"],), f_sb)

    # Calculate absorption (convert dB to Np: 1dB is 0.115Np)
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 14
    @. spl_sb *= exp(-2 * 0.115 * alpha * (r - settings["r_0"]))
end