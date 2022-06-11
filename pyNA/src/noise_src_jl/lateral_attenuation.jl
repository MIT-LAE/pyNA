function lateral_attenuation(settings, beta, x_obs)
    # Depression angle: phi_d = beta (elevation angle) + epsilon (aircraft bank angle = 0)
    # Note: beta is in degrees
    T = eltype(beta)
    phi_d = beta

    # Lateral side distance
    l = x_obs[2]

    # Engine installation term [dB]
    if settings.engine_mounting == "underwing"
        E_eng = 10 * log10.((0.0039 * cos.(phi_d*pi/180).^2 .+ sin.(phi_d*pi/180).^2).^0.062./(0.8786 * sin.(2 * phi_d*pi/180).^2 .+ cos.(2*phi_d*pi/180).^2))
    elseif settings.engine_mounting == "fuselage"
        E_eng = 10 * log10.((0.1225 * cos.(phi_d*pi/180).^2 .+ sin.(phi_d*pi/180).^2).^0.329)
    elseif settings.engine_mounting == "propeller"
        E_eng = zeros(size(beta))
    elseif settings.engine_mounting == "none"
        E_eng = zeros(size(beta))
    else
        throw(DomainError("Invalid engine_mounting specified. Specify: underwing/fuselage/propeller/none."))
    end

    # Attenuation caused by ground and refracting-scattering effects [dB]
    # Note: beta is in degrees
    A_grs = zeros(T, size(beta))
    A_grs[findall(beta.<=50.)] = (1.137 .- 0.0229*beta .+ 9.72 * exp.(-0.142 * beta))[findall(beta.<=50.)]
    
    # Over-ground attenuation [dB]
    if 0. <= l <= 914
        g = 11.83 * (1 - exp(-0.00274 * l))
    elseif l > 914
        g = 10.86  # 11.83*(1-exp(-0.00274*914))
    else
        throw(DomainError("Lateral sideline distance negative."))
    end

    # Overall lateral attenuation
    Lambda = 10 .^ ((E_eng .- g * A_grs ./ 10.86) ./ 10.)

    return Lambda
end