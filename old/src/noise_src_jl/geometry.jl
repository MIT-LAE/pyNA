using ReverseDiff

function compute_euler_transformation(n_vcr_a_1, n_vcr_a_2, n_vcr_a_3, theta_B)

    cth  = cos(deg2rad(theta_B))
    sth  = sin(deg2rad(theta_B))
    n_vcr_s_1 = cth * n_vcr_a_1 - sth * n_vcr_a_3
    n_vcr_s_2 = n_vcr_a_2
    n_vcr_s_3 = sth * n_vcr_a_1 + cth * n_vcr_a_3
    
    return n_vcr_s_1, n_vcr_s_2, n_vcr_s_3
end

function compute_average_speed_of_sound(z, c_0, T_0)

    n_intermediate = 0
    dz = z / (n_intermediate+1)
    c_bar = c_0

    # Step from altitude back down to sea level (sign is opposite for the atmospheric temperature formula)
    for k in 1:(n_intermediate+1)
        T_im = T_0 - k * dz * (-0.0065)
        c_im = sqrt(1.4 * 287. * T_im)
        c_bar = (k) / (k + 1) * c_bar + c_im / (k + 1)
    end

    return c_bar
end

function geometry!(y::Array, input_v::Union{Array, ReverseDiff.TrackedArray}, x_obs::Array{Float64, 1})

    # Extract inputs
    # input_v = [x, y, z, alpha, gamma, t_s, c_0, T_0]
    # y = [r, beta, theta, phi, c_bar, t_o]
    
    # Compute body angles (psi_B, theta_B, phi_B): angle of body w.r.t. horizontal
    # theta_B = alpha + gamma
    # phi_B = 0.
    # psi_B = 0.

    # Compute the relative observer-aircraft position vector i.e. difference between observer and ac coordinate
    # Note: add 4 meters to the alitude of the aircraft (for engine height)
    r_1 =  x_obs[1] -  input_v[1]
    r_2 =  x_obs[2] -  input_v[2]
    r_3 = -x_obs[3] + (input_v[3] + 4.)

    # Normalize the distance vector
    r = sqrt(r_1 ^2 + r_2 ^2 + r_3 ^2)
    n_vcr_a_1 = r_1 / r
    n_vcr_a_2 = r_2 / r
    n_vcr_a_3 = r_3 / r

    # Define elevation angle (with respect to the horizontal plane of the microphone)
    beta = asind(n_vcr_a_3)

    # Transformation direction cosines (Euler angles) to the source coordinate system (i.e. take position of the aircraft into account)
    n_vcr_s_1, n_vcr_s_2, n_vcr_s_3 = compute_euler_transformation(n_vcr_a_1, n_vcr_a_2, n_vcr_a_3, input_v[4] + input_v[5])

    # Define polar directivity angle
    theta = acosd(n_vcr_s_1)

    # Define azimuthal directivity angle
    phi = -atand(n_vcr_s_2, n_vcr_s_3)

    # Average speed of sound
    c_bar = compute_average_speed_of_sound(input_v[3], input_v[7], input_v[8])

    # Define observed time
    t_o = input_v[6] + r / c_bar

    y .= [r, beta, theta, phi, c_bar, t_o]

end

geometry_fwd! = (y, x)->geometry!(y, x, x_obs)