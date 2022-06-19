include("direct_propagation.jl")
include("atmospheric_absorption.jl")
include("lateral_attenuation.jl")
include("ground_reflections.jl")


function propagation!(spl, pyna_ip, settings, f_sb, x_obs, r, x, z, c_bar, rho_0, I_0, beta)

    if settings.direct_propagation
        direct_propagation!(spl, settings, r, I_0)
    end

    # Split mean-square acoustic pressure in frequency sub-bands
    if settings.N_b > 1
        spl_sb = split_subbands(settings, spl)
    else
        spl_sb = spl
    end

    # Apply atmospheric absorption on sub-bands
    z = max.(z, 0)
    if settings.absorption
        atmospheric_absorption!(spl_sb, pyna_ip, settings, f_sb, z, r)
    end

    # Apply ground effects on sub-bands
    if settings.groundeffects
        # Empirical lateral attenuation for microphone on sideline
        if (settings.lateral_attenuation == true) && (x_obs[2] != 0)
            # Lateral attenuation factor
            lateral_attenuation!(spl_sb, settings, beta, x_obs)
            
            # Ground reflection factor for center-line
            r_cl = sqrt((x-x_obs[1])^2 + 1^2 + z^2)
            beta_cl = asin(z/r_cl) * (180/ pi)
            ground_reflections!(spl_sb, pyna_ip, settings, f_sb, x_obs, r_cl, beta_cl, c_bar, rho_0)
        else
            ground_reflections!(spl_sb, pyna_ip, settings, f_sb, x_obs, r, beta, c_bar, rho_0)
        end
    end

    # Recombine the mean-square acoustic pressure in the frequency sub-bands
    if settings.N_b > 1
        spl_combined = reshape(sum(reshape(spl_sb, (settings.N_b, settings.N_f)), dims=1), (settings.N_f,))
    else
        spl_combined = spl
    end

    # Output
    @. spl = spl_combined
end