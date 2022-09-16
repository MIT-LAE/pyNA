function ground_effects!(spl_sb, x, settings, pyna_ip, f_sb, x_obs)

    # x = [r, beta, c_bar, rho_0]
    # y = spl_sb

    # Calculate difference in direct and reflected distance between source and observer
    # Source: Berton - lateral attenuation paper (2019)
    r_r = sqrt(x[1]^2 + 4 * x_obs[3]^2 + 4 * x[1] * x_obs[3] * sin(x[2]*π/180))
    dr = r_r - x[1]

    # Source: Zorumski report 1982 part 1. Chapter 3.2 page 1
    k = 2 * π * f_sb ./ x[3]

    # Calculate dimensionless frequency eta (note: for acoustically hard surface: eta = 0)
    # Source: Zorumski report 1982 part 1. Chapter 3.2 page 2
    eta = 2 * π * x[4] * f_sb / settings["ground_resistance"]

    # Calculate the cosine of the incidence angle
    cos_theta = (x[1] * sin(x[2] * π / 180.) + 2 * x_obs[3]) / r_r

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
    for i in 1:1:settings["n_frequency_subbands"]*settings["n_frequency_bands"]
        
        # Unit step function
        if - tau_re[i] > 0
            U = 1.0
        elseif - tau_re[i] == 0
            U = 0.5
        else
            U = 0.
        end

        # Calculate F
        # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 11
        tau_sq_re = (tau_re[i]^2 + tau_im[i]^2) * (cos(2 * atan(tau_im[i]/tau_re[i])))
        tau_sq_im = (tau_re[i]^2 + tau_im[i]^2) * (sin(2 * atan(tau_im[i]/tau_re[i])))

        tau_4th_re = (tau_sq_re^2 + tau_sq_im^2) * (cos(2 * atan(tau_sq_im/tau_sq_re)))
        tau_4th_im = (tau_sq_re^2 + tau_sq_im^2) * (sin(2 * atan(tau_sq_im/tau_sq_re)))
        term1_re = -2 * sqrt(π) * U * (tau_re[i] * (exp(tau_sq_re) * cos(tau_sq_im)) - tau_im[i] * (exp(tau_sq_re) * sin(tau_sq_im)))
        term1_im = -2 * sqrt(π) * U * (tau_im[i] * (exp(tau_sq_re) * cos(tau_sq_im)) + tau_re[i] * (exp(tau_sq_re) * sin(tau_sq_im)))
        
        tau_absolute = sqrt(tau_re[i]^2 + tau_im[i]^2)
        
        if tau_absolute >= 10
            F_re = term1_re + 1/2. * ( tau_sq_re / (tau_sq_re^2 + tau_sq_im^2)) + (-3/4 * ( tau_4th_re / (tau_4th_re^2 + tau_4th_im^2)))
            F_im = term1_im + 1/2. * (-tau_sq_im / (tau_sq_re^2 + tau_sq_im^2)) + (-3/4 * (-tau_4th_im / (tau_4th_re^2 + tau_4th_im^2)))
        else
            # Correct F for small tau
            itau_re = -tau_im[i]
            itau_im =  tau_re[i]
            fad_re = pyna_ip.f_faddeeva_real(itau_im, itau_re)
            fad_im = pyna_ip.f_faddeeva_imag(itau_im, itau_re)
            F_re = 1 - sqrt(π) * (tau_re[i] * fad_re - tau_im[i] * fad_im)
            F_im =   - sqrt(π) * (tau_im[i] * fad_re + tau_re[i] * fad_im) 
        end

        # Calculate Z_cswfc
        # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 6
        Z_cswfc_re = Gamma_re[i] + ((1 - Gamma_re[i]) * F_re - (   - Gamma_im[i]) * F_im)
        Z_cswfc_im = Gamma_im[i] + ((  - Gamma_im[i]) * F_re + (1. - Gamma_re[i]) * F_im)

        # Calculate R and alpha
        R = sqrt(Z_cswfc_re^2 + Z_cswfc_im^2)
        alpha = 2 * atan((R - Z_cswfc_re) / Z_cswfc_im)

        # Calculate the constant K and constant epsilon
        # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 16-17
        K = 2^(1 / (6 * settings["n_frequency_subbands"]))
        eps = K - 1

        # Multiply subbands with G
        # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 18
        if dr == 0
            spl_sb[i] *= (1. + R^2 + 2 * R * exp(-(settings["incoherence_constant"] * k[i] * dr)^2) * cos(alpha + k[i] * dr))
        else
            spl_sb[i] *= (1. + R^2 + 2 * R * exp(-(settings["incoherence_constant"] * k[i] * dr)^2) * cos(alpha + k[i] * dr) * sin(eps * k[i] * dr) / (eps * k[i] * dr))
        end

    end

end

ground_effects_fwd! = (y,x)->ground_effects!(y, x, settings, pyna_ip, f_sb, x_obs)