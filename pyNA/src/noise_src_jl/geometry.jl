function compute_average_speed_of_sound(settings, z, c_0, T_0)

    n_intermediate = 11
    dz = z ./ (n_intermediate-1)
    c_bar = c_0
    for k in 1:(n_intermediate-1)
        T_im = T_0 .+ settings.dT .- (z .- k .* dz) .* (-0.0065)
        c_im = sqrt.(1.4 * 287. .* T_im)
        c_bar = (k) / (k + 1) .* c_bar .+ c_im ./ (k + 1)
    end

    return c_bar
end

function compute_euler_transformation(n_vcr_a_1, n_vcr_a_2, n_vcr_a_3, theta_B, phi_B, psi_B)

    cth  = cos.(deg2rad.(theta_B))
    sth  = sin.(deg2rad.(theta_B))
    cphi = cos.(deg2rad.(phi_B))
    sphi = sin.(deg2rad.(phi_B))
    cpsi = cos.(deg2rad.(psi_B))
    spsi = sin.(deg2rad.(psi_B))
    n_vcr_s_1 = cth .* cpsi .* n_vcr_a_1 .+ cth .* spsi .* n_vcr_a_2 - sth .* n_vcr_a_3
    n_vcr_s_2 = (-spsi .* cphi .+ sphi .* sth .* cpsi) .* n_vcr_a_1 .+ ( cphi .* cpsi .+ sphi .* sth .* spsi) .* n_vcr_a_2 .+ sphi .* cth .* n_vcr_a_3
    n_vcr_s_3 = (spsi .* sphi .+ cphi .* sth .* cpsi) .* n_vcr_a_1 .+ ( -sphi .* cpsi .+ cphi .* sth .* spsi) .* n_vcr_a_2 .+ cphi .* cth .* n_vcr_a_3
    
    return n_vcr_s_1, n_vcr_s_2, n_vcr_s_3
end 

function geometry(settings, x_obs::Array{Float64, 1}, x, y, z, alpha, gamma, t_s, c_0, T_0)
    
    # Size of inputs
    n_t = size(x)[1]

    # Initialize vectors
    T = eltype(x)
    r = zeros(T, n_t)
    beta = zeros(T, n_t)
    theta = zeros(T, n_t)
    phi = zeros(T, n_t)
    c_bar = zeros(T, n_t)
    t_o = zeros(T, n_t)

    # Step through all time steps
    for i in 1:n_t

        # Compute body angles (psi_B, theta_B, phi_B): angle of body w.r.t. horizontal
        theta_B = alpha[i] .+ gamma[i]
        phi_B = 0.
        psi_B = 0.

        # Compute the relative observer-aircraft position vector i.e. difference between observer and ac coordinate
        r_1 =  x_obs[1] .- x[i]
        r_2 =  x_obs[2] .- y[i]
        r_3 = -x_obs[3] .+ z[i]

        # Normalize the distance vector
        r[i] = sqrt.(r_1 .^2 .+ r_2 .^2 .+ r_3 .^2)
        n_vcr_a_1 = r_1 ./ r[i]
        n_vcr_a_2 = r_2 ./ r[i]
        n_vcr_a_3 = r_3 ./ r[i]

        # Define elevation angle
        beta[i] = asind.(n_vcr_a_3)

        # Transformation direction cosines (Euler angles) to the source coordinate system (i.e. take position of the aircraft into account)
        n_vcr_s_1, n_vcr_s_2, n_vcr_s_3 = compute_euler_transformation(n_vcr_a_1, n_vcr_a_2, n_vcr_a_3, theta_B, phi_B, psi_B)

        # Define polar directivity angle
        theta[i] = acosd.(n_vcr_s_1)

        # Define azimuthal directivity angle
        phi[i] = -atand.(n_vcr_s_2, n_vcr_s_3)

        # Average speed of sound
        c_bar[i] = compute_average_speed_of_sound(settings, z[i], c_0[i], T_0[i])

        # Define observed time
        t_o[i] = t_s[i] .+ r[i] ./ c_bar[i]
    end    

    return r, beta, theta, phi, c_bar, t_o
end