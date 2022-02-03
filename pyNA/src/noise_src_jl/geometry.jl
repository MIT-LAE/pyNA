function geometry(settings, x_obs, n_t, input_geom)
    # Unpack inputs
    x = input_geom[0*n_t + 1 : 1*n_t]
    y = input_geom[1*n_t + 1 : 2*n_t]
    z = input_geom[2*n_t + 1 : 3*n_t]
    alpha = input_geom[3*n_t + 1 : 4*n_t]
    gamma = input_geom[4*n_t + 1 : 5*n_t]
    t_s = input_geom[5*n_t + 1 : 6*n_t]
    c_0 = input_geom[6*n_t + 1 : 7*n_t]
    T_0 = input_geom[7*n_t + 1 : 8*n_t]
    
    # Compute body angles (psi_B, theta_B, phi_B): angle of body w.r.t. horizontal
    theta_B = alpha .+ gamma
    phi_B = zeros(Float64, n_t)
    psi_B = zeros(Float64, n_t)
    
    # Compute the relative observer-aircraft position vector i.e. difference between observer and ac coordinate
    r_1 =  x_obs[1] .- x
    r_2 =  x_obs[2] .- y
    r_3 = -x_obs[3] .+ z
    
    # Normalize the distance vector
    r = sqrt.(r_1 .^2 .+ r_2 .^2 .+ r_3 .^2)
    
    n_vcr_a_1 = r_1 ./ r
    n_vcr_a_2 = r_2 ./ r
    n_vcr_a_3 = r_3 ./ r

    # Define elevation angle
    beta = 180. / pi .* asin.(n_vcr_a_3)

    # Transformation direction cosines (Euler angles) to the source coordinate system (i.e. take position of the aircraft into account)
    cth  = cos.(pi / 180. .* theta_B)
    sth  = sin.(pi / 180. .* theta_B)
    cphi = cos.(pi / 180. .* phi_B)
    sphi = sin.(pi / 180. .* phi_B)
    cpsi = cos.(pi / 180. .* psi_B)
    spsi = sin.(pi / 180. .* psi_B)
    n_vcr_s_1 = cth .* cpsi .* n_vcr_a_1 .+ cth .* spsi .* n_vcr_a_2 - sth .* n_vcr_a_3
    n_vcr_s_2 = (-spsi .* cphi .+ sphi .* sth .* cpsi) .* n_vcr_a_1 .+ ( cphi .* cpsi .+ sphi .* sth .* spsi) .* n_vcr_a_2 .+ sphi .* cth .* n_vcr_a_3
    n_vcr_s_3 = (spsi .* sphi .+ cphi .* sth .* cpsi) .* n_vcr_a_1 .+ ( -sphi .* cpsi .+ cphi .* sth .* spsi) .* n_vcr_a_2 .+ cphi .* cth .* n_vcr_a_3

    # Define polar directivity angle
    theta = 180. / pi .* acos.(n_vcr_s_1)

    # Define azimuthal directivity angle
    phi = -180. / pi .* atan.(n_vcr_s_2, n_vcr_s_3)

    # Average speed of sound: CHECK
    n_intermediate = 11
    dz = z ./ (n_intermediate-1)
    c_bar = c_0
    for k in range(1, n_intermediate-1, step=1)
        T_im = T_0 .+ settings.dT .- (z .- k .* dz) .* (-0.0065)  # Temperature
        c_im = sqrt.(1.4 * 287. .* T_im)  # Speed of sound
        c_bar = (k) / (k + 1) .* c_bar .+ c_im ./ (k + 1)
    end

    # Define observed time
    t_o = t_s .+ r ./ c_bar

    # Create output
    output_geom = r
    output_geom = vcat(output_geom, beta)
    output_geom = vcat(output_geom, theta)
    output_geom = vcat(output_geom, phi)
    output_geom = vcat(output_geom, c_bar)
    output_geom = vcat(output_geom, t_o)

    return output_geom
end