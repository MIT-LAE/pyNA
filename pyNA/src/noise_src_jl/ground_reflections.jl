function ground_reflections(settings::PyObject, data::PyObject, x_obs::Array{Float64}, r, beta, c_bar, rho_0)

    # Size of trajectory
    n_t = size(r)[1]

    # Initialize solution matrix
    G = zeros(eltype(r), (n_t, settings.N_b*settings.N_f))

    # Step through all time steps
    for i in 1:n_t

        # Calculate difference in direct and reflected distance between source and observer
        # Source: Berton - lateral attenuation paper (2019)
        r_r = sqrt(r[i]^2 + 4 * x_obs[3]^2 + 4 * r[i] * x_obs[3] * sin(beta[i]*π/180.))
        dr = r_r - r[i]

        for j in 1:settings.N_f*settings.N_b

            # Source: Zorumski report 1982 part 1. Chapter 3.2 page 1
            k = 2 * π * data.f_sb[j] ./ c_bar[i]
            
            # Calculate dimensionless frequency eta (note: for acoustically hard surface: eta = 0)
            # Source: Zorumski report 1982 part 1. Chapter 3.2 page 2
            eta = 2 * π * rho_0[i] .* data.f_sb[j] / settings.sigma

            # Calculate the cosine of the incidence angle
            cos_theta = (r[i] * sin(beta[i] * π / 180.) + 2 * x_obs[3]) / r_r

            # Complex specific ground admittance nu
            # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 13 / adapted through Berton lateral attenuation paper
            nu_re = (1 + (6.86 * eta)^(-0.75)) / ((1 + (6.86 * eta)^(-0.75))^2 + ((4.36 * eta)^(-0.73))^2)
            nu_im = -((4.36 * eta)^(-0.73)) / ((1 + (6.86 * eta)^(-0.75))^2 + ((4.36 * eta)^(-0.73))^2)
            
            # Calculate Gamma
            # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 5
            Gamma_re = (cos_theta - nu_re) * ((cos_theta + nu_re) / ((cos_theta + nu_re)^2 + nu_im^2)) - (-nu_im) * (-nu_im / ((cos_theta + nu_re)^2 + nu_im^2))
            Gamma_im = (-nu_im) * ((cos_theta + nu_re) / ((cos_theta + nu_re)^2 + nu_im^2)) + (cos_theta - nu_re) * (-nu_im / ((cos_theta + nu_re)^2 + nu_im^2))

            # Calculate tau
            # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 9
            tau_re = (sqrt.(k * r_r / 2) * cos(-π / 4.)) * (cos_theta + nu_re) .- (sqrt(k * r_r / 2) * sin(-π / 4.)) * nu_im
            tau_im = (sqrt.(k * r_r / 2) * sin(-π / 4.)) * (cos_theta + nu_re) .+ (sqrt(k * r_r / 2) * cos(-π / 4.)) * nu_im

            # Calculate complex spherical wave reflection coefficient
            # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 12    
            if -tau_re > 0
                U = tau_re.^0
            elseif -tau_re == 0
                U = 0.5*tau_re.^0
            else
                U = tau_re.^0 - 1
            end
            
            # Calculate F
            # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 11
            tau_sq_re = (tau_re^2 + tau_im^2) * (cos(2 * atan(tau_im / tau_re)))
            tau_sq_im = (tau_re^2 + tau_im^2) * (sin(2 * atan(tau_im / tau_re)))
            tau_4th_re = (tau_sq_re^2 + tau_sq_im.^2) * (cos(2 * atan(tau_sq_im / tau_sq_re)))
            tau_4th_im = (tau_sq_re^2 + tau_sq_im.^2) * (sin(2 * atan(tau_sq_im / tau_sq_re)))
            term1_re = -2 * sqrt(π) * U * (tau_re * (exp.(tau_sq_re) * cos.(tau_sq_im)) - tau_im * (exp(tau_sq_re) * sin(tau_sq_im)))
            term1_im = -2 * sqrt(π) * U * (tau_im * (exp.(tau_sq_re) * cos.(tau_sq_im)) + tau_re * (exp(tau_sq_re) * sin(tau_sq_im)))

            F_re = term1_re + 1/2. * ( tau_sq_re / (tau_sq_re^2 + tau_sq_im^2)) + (-3/4 * ( tau_4th_re / (tau_4th_re^2 + tau_4th_im^2)))
            F_im = term1_im + 1/2. * (-tau_sq_im / (tau_sq_re^2 + tau_sq_im^2)) + (-3/4 * (-tau_4th_im / (tau_4th_re^2 + tau_4th_im^2)))

            # Correct F for small tau
            itau_re = -tau_im
            itau_im = tau_re 

            f_faddeeva_real = LinearInterpolation((data.Faddeeva_itau_im, data.Faddeeva_itau_re), data.Faddeeva_real, extrapolation_bc=Flat())
            f_faddeeva_imag = LinearInterpolation((data.Faddeeva_itau_im, data.Faddeeva_itau_re), data.Faddeeva_imag, extrapolation_bc=Flat())
            fad_re = f_faddeeva_real(itau_im, itau_re)
            fad_im = f_faddeeva_imag(itau_im, itau_re)
            F_correction_re = 1 - sqrt(π) * (tau_re * fad_re - tau_im * fad_im)
            F_correction_im =   - sqrt(π) * (tau_im * fad_re + tau_re * fad_im)
            tau_absolute = sqrt(tau_re^2 .+ tau_im^2)

            if tau_absolute < 10
                F_re = F_correction_re
                F_im = F_correction_im
            end

            # Calculate Z_cswfc
            # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 6
            Z_cswfc_re = Gamma_re + ((1 - Gamma_re) * F_re - (   - Gamma_im) * F_im)
            Z_cswfc_im = Gamma_im + ((  - Gamma_im) * F_re + (1. - Gamma_re) * F_im)

            # Calculate R and alpha
            R = sqrt(Z_cswfc_re.^2 + Z_cswfc_im^2)
            alpha = 2 * atan((R - Z_cswfc_re) / Z_cswfc_im)

            # Calculate the constant K and constant epsilon
            # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 16-17
            K = 2^(1 / (6 * settings.N_b))
            eps = K - 1

            # Calculate G
            # Source: Zorumski report 1982 part 1. Chapter 3.2 Equation 18
            if dr == 0
                G[i,j] = 1 + R^2 + 2 * R * exp(-(settings.a_coh * k * dr)^2) * cos(alpha + k * dr)
            else
                G[i,j] = 1 + R^2 + 2 * R * exp(-(settings.a_coh * k * dr)^2) * cos(alpha + k * dr) * sin(eps * k * dr) / (eps * k * dr)
            end
        
        end
    end
    
    return G
end