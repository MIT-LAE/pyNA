function propagation(settings::PyObject, data::PyObject, x_obs::Array{Float64, 1}, msap_source, r, x, z, c_bar, rho_0, I_0, beta)

    # Impedance at the observer
    I_0_obs = 409.74

    # Size of inputs
    n_t = size(r)[1]

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
        msap_sb = split_subbands(settings, msap_direct_prop)
    else
        msap_sb = msap_direct_prop
    end

    # Apply atmospheric absorption on sub-bands
    if settings.absorption
        # Calculate average absorption factor between observer and source
        f_abs = LinearInterpolation((data.abs_alt, data.abs_freq), data.abs)
        alpha_f = f_abs.(z.*ones(eltype(r), (1, settings.N_b*settings.N_f)), reshape(data.f_sb, (1,settings.N_f*settings.N_b)).*ones(eltype(r), (n_t,1)))

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
            G = ground_reflections(settings, data, x_obs, r_cl, beta_cl, c_bar, rho_0)

            # Apply ground effects
            msap_ge = msap_abs .* (G .* Lambda)
        else
            # Compute ground effects factor
            G = ground_reflections(settings, data, x_obs, r, beta, c_bar, rho_0)
            
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

    println("End propagation")

    return msap_prop
end