include("direct_propagation.jl")
include("atmospheric_absorption.jl")
include("lateral_attenuation.jl")
include("ground_effects.jl")


function propagation!(spl, pyna_ip, settings, f_sb, x_obs, r, x, z, c_bar, rho_0, I_0, beta)

    if settings["direct_propagation"]
        direct_propagation!(spl, settings, r, I_0)
    end

    # Split mean-square acoustic pressure in frequency sub-bands
    if settings["n_frequency_subbands"] > 1
        spl_sb = split_subbands(settings, spl)
    else
        spl_sb = spl
    end

    # Apply atmospheric absorption on sub-bands
    z = max.(z, 0)
    if settings["absorption"]
        atmospheric_absorption!(spl_sb, pyna_ip, settings, f_sb, z, r)
    end

    # Apply ground effects on sub-bands
    if settings["ground_effects"]
        ground_effects!(spl_sb, pyna_ip, settings, f_sb, x_obs, r, beta, c_bar, rho_0)
    end

    # Recombine the mean-square acoustic pressure in the frequency sub-bands
    if settings["n_frequency_subbands"] > 1
        spl_combined = reshape(sum(reshape(spl_sb, (settings["n_frequency_subbands"], settings["n_frequency_bands"])), dims=1), (settings["n_frequency_bands"],))
    else
        spl_combined = spl
    end

    # Output
    @. spl = spl_combined
end