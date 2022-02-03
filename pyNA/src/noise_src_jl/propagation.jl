function propagation(settings, data, x_obs, n_t, msap_source, input_prop)

    # Extract inputs and outputs
    r = input_prop[0*n_t + 1 : 1*n_t]
    x = input_prop[1*n_t + 1 : 2*n_t]
    z = input_prop[2*n_t + 1 : 3*n_t]
    c_bar = input_prop[3*n_t + 1 : 4*n_t]
    rho_0 = input_prop[4*n_t + 1 : 5*n_t]
    I_0 = input_prop[5*n_t + 1 : 6*n_t]
    I_0_obs = 409.74
    beta = input_prop[6*n_t + 1 : 7*n_t]

    # Get type of input variables
    T = eltype(msap_source)
    msap_prop = zeros(T, (n_t, settings.N_f))

    # Apply spherical spreading and characteristic impedance effects to the MSAP
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 1
    z = max.(z, 0)
    if settings.direct_propagation
        msap_direct_prop = msap_source .* ((settings.r_0^2 ./ r.^2) .* (I_0_obs./I_0))
    else
        msap_direct_prop = msap_source
    end
    
    # Split mean-square acoustic pressure in frequency sub-bands
    if settings.N_b > 1
        msap_sb = split_subbands(settings, n_t, msap_direct_prop)
    else
        msap_sb = msap_direct_prop
    end

    # Apply atmospheric absorption on sub-bands
    if settings.absorption
        # Calculate average absorption factor between observer and source
        f_abs = LinearInterpolation((data.abs_alt, data.abs_freq), data.abs)
        alpha_f = f_abs.(z.*ones(eltype(input_prop), (1, settings.N_b*settings.N_f)), reshape(data.f_sb, (1,settings.N_f*settings.N_b)).*ones(eltype(input_prop), (n_t,1)))

        # Calculate absorption (convert dB to Np: 1dB is 0.115Np)
        # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 14
        msap_abs = msap_sb .* exp.(-2 * 0.115 * alpha_f .* (r .- settings.r_0))
    else
        msap_abs = msap_direct_prop
    end

    # Apply ground effects on sub-bands
    if settings.groundeffects
        # Empirical lateral attenuation for microphone on sideline
        if (settings.lateral_attenuation == true) && (x_obs[2] != 0)
            # Lateral attenuation factor
            Lambda = lateral_attenuation(settings, beta, x_obs)
            
            # Compute elevation angle from "center-line" observer. 
            r_cl = sqrt.((x.-x_obs[1]).^2 .+ 1. ^2 .+ z.^2)
            beta_cl = asin.(z./r_cl) * 180. / pi

            # Ground reflection factor for center-line
            G = ground_reflections(settings, data, n_t, r_cl, beta_cl, x_obs, c_bar, rho_0)

            # Apply ground effects
            msap_ge = msap_abs .* (G .* Lambda)
        else
            # Compute ground effects factor
            G = ground_reflections(settings, data, n_t, r, beta, x_obs, c_bar, rho_0)
            
            # Multiply subbands with G
            msap_ge = msap_abs .* G
        end
    else
        msap_ge = msap_abs
    end

    # Recombine the mean-square acoustic pressure in the frequency sub-bands
    if settings.N_b > 1
        msap_prop = transpose(reshape(sum(reshape(transpose(msap_ge), (settings.N_b, settings.N_f, n_t)), dims=1), (settings.N_f, n_t)))        
    else
        msap_prop = msap_ge
    end

    return msap_prop
end

