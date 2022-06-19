function ground_reflections!(spl_sb, pyna_ip, settings, f_sb, x_obs, r, beta, c_bar, rho_0)

    # Calculate difference in direct and reflected distance between source and observer
    # Source: Berton - lateral attenuation paper (2019)
    r_r = sqrt(r^2 + 4 * x_obs[3]^2 + 4 * r * x_obs[3] * sin(beta*π/180))
    dr = r_r - r

    # Source: Zorumski report 1982 part 1. Chapter 3.2 page 1
    k = 2 * π * f_sb ./ c_bar

    # Calculate dimensionless frequency eta (note: for acoustically hard surface: eta = 0)
    # Source: Zorumski report 1982 part 1. Chapter 3.2 page 2
    eta = 2 * π * rho_0 * f_sb / settings.sigma 

    # Calculate the cosine of the incidence angle
    cos_theta = (r * sin(beta * π / 180.) + 2 * x_obs[3]) / r_r

    # Complex specific ground admittance nu
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 13 / adapted through Berton lateral attenuation paper
    nu_re = (1 .+ (6.86 * eta).^(-0.75)) ./ ((1 .+ (6.86 * eta).^(-0.75)).^2 .+ ((4.36 * eta).^(-0.73)).^2)
    nu_im = -((4.36 * eta).^(-0.73)) ./ ((1 .+ (6.86 * eta).^(-0.75)).^2 .+ ((4.36 * eta).^(-0.73)).^2)

    # Calculate Gamma
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 5
    Gamma_re = (cos_theta .- nu_re) .* ((cos_theta .+ nu_re) ./ ((cos_theta .+ nu_re).^2 .+ nu_im.^2)) .- (-nu_im) .* (-nu_im ./ ((cos_theta .+ nu_re).^2 .+ nu_im.^2))
    Gamma_im = (-nu_im) .* ((cos_theta .+ nu_re) ./ ((cos_theta .+ nu_re).^2 .+ nu_im.^2)) .+ (cos_theta .- nu_re) .* (-nu_im ./ ((cos_theta .+ nu_re).^2 .+ nu_im.^2))

    # Calculate tau
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 9
    tau_re = (sqrt.(k .* r_r / 2) * cos(-π / 4.)) .* (cos_theta .+ nu_re) .- (sqrt.(k .* r_r / 2) * sin(-π / 4.)) .* nu_im
    tau_im = (sqrt.(k .* r_r / 2) * sin(-π / 4.)) .* (cos_theta .+ nu_re) .+ (sqrt.(k .* r_r / 2) * cos(-π / 4.)) .* nu_im

    # Calculate complex spherical wave reflection coefficient
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 12    
    U = zeros(eltype(r), (settings.N_b*settings.N_f,))
    U[findall(-tau_re.>0)] = (tau_re.^0)[findall(-tau_re.>0)]
    U[findall(-tau_re .== 0)] = 0.5*(tau_re.^0)[findall(-tau_re .== 0)]

    # Calculate F
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 11
    tau_sq_re = (tau_re.^2 + tau_im.^2) .* (cos.(2 * atan.(tau_im ./ tau_re)))
    tau_sq_im = (tau_re.^2 + tau_im.^2) .* (sin.(2 * atan.(tau_im ./ tau_re)))
    tau_4th_re = (tau_sq_re.^2 .+ tau_sq_im.^2) .* (cos.(2 * atan.(tau_sq_im ./ tau_sq_re)))
    tau_4th_im = (tau_sq_re.^2 .+ tau_sq_im.^2) .* (sin.(2 * atan.(tau_sq_im ./ tau_sq_re)))
    term1_re = -2 * sqrt(π) * U .* (tau_re .* (exp.(tau_sq_re) .* cos.(tau_sq_im)) .- tau_im .* (exp.(tau_sq_re) .* sin.(tau_sq_im)))
    term1_im = -2 * sqrt(π) * U .* (tau_im .* (exp.(tau_sq_re) .* cos.(tau_sq_im)) .+ tau_re .* (exp.(tau_sq_re) .* sin.(tau_sq_im)))

    F_re = term1_re .+ 1/2. .* ( tau_sq_re ./ (tau_sq_re.^2 .+ tau_sq_im.^2)) .+ (-3/4 .* ( tau_4th_re ./ (tau_4th_re.^2 + tau_4th_im.^2)))
    F_im = term1_im .+ 1/2. .* (-tau_sq_im ./ (tau_sq_re.^2 .+ tau_sq_im.^2)) .+ (-3/4 .* (-tau_4th_im ./ (tau_4th_re.^2 + tau_4th_im.^2)))

    # Correct F for small tau
    itau_re = -tau_im
    itau_im = tau_re 

    fad_re = pyna_ip.f_faddeeva_real.(itau_im, itau_re)
    fad_im = pyna_ip.f_faddeeva_imag.(itau_im, itau_re)
    F_correction_re = 1 .- sqrt(π) .* (tau_re .* fad_re .- tau_im .* fad_im)
    F_correction_im =    - sqrt(π) .* (tau_im .* fad_re .+ tau_re .* fad_im)
    tau_absolute = sqrt.(tau_re.^2 .+ tau_im.^2)

    F_re[findall(tau_absolute .< 10.)] = F_correction_re[findall(tau_absolute .< 10)]
    F_im[findall(tau_absolute .< 10.)] = F_correction_im[findall(tau_absolute .< 10)]

    # Calculate Z_cswfc
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 6
    Z_cswfc_re = Gamma_re .+ ((1 .- Gamma_re) .* F_re .- (    - Gamma_im) .* F_im)
    Z_cswfc_im = Gamma_im .+ ((   - Gamma_im) .* F_re .+ (1. .- Gamma_re) .* F_im)

    # Calculate R and alpha
    R = sqrt.(Z_cswfc_re.^2 .+ Z_cswfc_im.^2)
    alpha = 2 * atan.((R - Z_cswfc_re) ./ Z_cswfc_im)

    # Calculate the constant K and constant epsilon
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 16-17
    K = 2^(1 / (6 * settings.N_b))
    eps = K - 1

    # Multiply subbands with G
    # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 18
    if dr == 0
        @. spl_sb *= (1. + R^2 + 2 * R * exp(-(settings.a_coh * k * dr)^2) * cos(alpha + k * dr))
    else
        @. spl_sb *= (1. + R^2 + 2 * R * exp(-(settings.a_coh * k * dr)^2) * cos(alpha + k * dr) * sin(eps * k * dr) / (eps * k * dr))
    end
end