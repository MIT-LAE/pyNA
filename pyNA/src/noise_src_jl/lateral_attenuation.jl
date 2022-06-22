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