function get_subband_frequencies!(f_sb, f, settings)

    # Calculate subband frequencies [Hz]
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 6-7
    # Source: Berton 2021 Simultaneous use of Ground Reflection and Lateral Attenuation Noise Models Appendix A Eq. 1
    
    m = (settings["n_frequency_subbands"] - 1) / 2.
    w = 2. ^ (1 / (3. * settings["n_frequency_subbands"]))

    for k in 1:1:settings["n_frequency_bands"]
        for h in 1:1:settings["n_frequency_subbands"]
            f_sb[(k-1)*settings["n_frequency_subbands"] + h] = w ^ (h - 1 - m) * f[k]
        end
    end

end