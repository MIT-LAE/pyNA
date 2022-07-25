function lateral_attenuation!(spl_sb, settings, beta, x_obs)
    
    # Engine installation term [dB]
    if settings.engine_mounting == "underwing"
        E_eng = 10 * log10((0.0039 * cos(beta*pi/180)^2 + sin(beta*pi/180)^2)^0.062/(0.8786 * sin(2 * beta*pi/180)^2 + cos(2*beta*pi/180)^2))
    elseif settings.engine_mounting == "fuselage"
        E_eng = 10 * log10((0.1225 * cos(beta*pi/180)^2 + sin(beta*pi/180)^2)^0.329)
    elseif settings.engine_mounting == "propeller"
        E_eng = 0.
    elseif settings.engine_mounting == "none"
        E_eng = 0.
    end

    # Attenuation caused by ground and refracting-scattering effects [dB]
    # Note: beta is in degrees
    if beta < 50.
        A_grs = 1.137 - 0.0229*beta + 9.72*exp(-0.142*beta)
    elseif beta < 0.
        throw(DomainError(beta, "beta is outside the valid domain (beta > 0)"))
    else
        A_grs = 0.
    end 

    # Over-ground attenuation [dB]
    if x_obs[2] <= 914
        g = 11.83 * (1 - exp(-0.00274 * x_obs[2]))
    else
        g = 10.86  # 11.83*(1-exp(-0.00274*914))
    end

    # Overall lateral attenuation
    @. spl_sb *= 10^((E_eng-g*A_grs/10.86)/10)

end