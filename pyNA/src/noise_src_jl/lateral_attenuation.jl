function lateral_attenuation(settings, x_obs, x, y, z)

    # Compute elevation angle (with respect to the horizontal plane of the microphone) - similar to geometry module
    r_1 =  x_obs[1] .- x
    r_2 =  x_obs[2] .- y
    r_3 = -x_obs[3] .+ (z .+ 4.)
    r = sqrt.(r_1.^2 + r_2.^2 + r_3.^2)
    n_vcr_a_3 = r_3 ./ r
    beta = asin.(n_vcr_a_3)*180/pi

    # Compute maximum elevation angle seen by the microphone using smooth maximum
    k_smooth = 50
    beta_max = smooth_max(k_smooth, beta)

    # Engine installation term [dB]
    if settings.engine_mounting == "underwing"
        E_eng = 10 * log10((0.0039 * cos(beta_max*pi/180)^2 + sin(beta_max*pi/180)^2)^0.062/(0.8786 * sin(2 * beta_max*pi/180)^2 + cos(2*beta_max*pi/180)^2))
    elseif settings.engine_mounting == "fuselage"
        E_eng = 10 * log10((0.1225 * cos(beta_max*pi/180)^2 + sin(beta_max*pi/180)^2)^0.329)
    elseif settings.engine_mounting == "propeller"
        E_eng = 0.
    elseif settings.engine_mounting == "none"
        E_eng = 0.
    end

    # Attenuation caused by ground and refracting-scattering effects [dB] (Note: beta is in degrees)
    if beta_max < 50.
        A_grs = 1.137 - 0.0229*beta_max + 9.72*exp(-0.142*beta_max)
    elseif beta_max < 0.
        throw(DomainError(beta_max, "beta_max is outside the valid domain (beta_max > 0)"))
    else
        A_grs = 0.
    end 

    # Over-ground attenuation [dB]
    if 0 <= x_obs[2] <= 914
        g = 11.83 * (1 - exp(-0.00274 * x_obs[2]))
    elseif x_obs[2] < 0.
        throw(DomainError(x_obs[2], "observer lateral position is outside the valid domain (x_obs[2] > 0)"))
    else
        g = 10.86  # 11.83*(1-exp(-0.00274*914))
    end

    # Overall lateral attenuation
    return delta_dB_lat = E_eng-g*A_grs/10.86

end

function lateral_attenuation!(spl_sb, settings, beta, x_obs)
    
    # Engine installation term [dB]
    if settings.engine_mounting == "underwing"
        E_eng = 10 * log10((0.0039 * cos(beta*pi/180)^2 + sin(beta*pi/180)^2)^0.062/(0.8786 * sin(2 * beta*pi/180)^2 + cos(2*beta*pi/180)^2))
    elseif settings.engine_mounting == "fuselage"
        E_eng = 10 * log10((0.1225 * cos(beta*pi/180)^2 + sin(beta*pi/180)^2)^0.329)
    elseif settings.engine_mounting == "propeller"
        E_eng = zeros(size(beta))
    elseif settings.engine_mounting == "none"
        E_eng = zeros(size(beta))
    end

    # Attenuation caused by ground and refracting-scattering effects [dB]
    # Note: beta is in degrees
    if beta < 50.
        A_grs = 1.137 - 0.0229*beta + 9.72*exp(-0.142*beta)
    else
        A_grs = 0.
    end 

    # Over-ground attenuation [dB]
    if 0. <= x_obs[2] <= 914
        g = 11.83 * (1 - exp(-0.00274 * x_obs[2]))
    elseif x_obs[2] > 914
        g = 10.86  # 11.83*(1-exp(-0.00274*914))
    else
        throw(DomainError("Lateral sideline distance negative."))
    end

    # Overall lateral attenuation
    @. spl_sb *= 10^((E_eng-g*A_grs/10.86)/10)

end