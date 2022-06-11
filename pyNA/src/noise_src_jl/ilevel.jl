using PCHIPInterpolation

function f_ilevel(t_o, level)

    # Number of time steps
    n_t = size(t_o)[1]

    # Interpolate time, level
    dt = 0.5
    n_ip = Int64(ceil((t_o[end]-t_o[1])/dt))

    T = eltype(level)
    t_ip = zeros(T, n_ip)
    for i in range(1, n_ip, step=1)
        t_ip[i] = t_o[1] + (i-1)*dt
    end

    # Interpolate the data
    f_ilevel_interp = PCHIPInterpolation.Interpolator(t_o, reshape(level, (n_t, )))
    level_ip = f_ilevel_interp.(t_ip)
        
    # Compute max. level
    level_max = maximum(level_ip)

    # ICAO Annex 16 procedures (p132 - App. 2-18)
    f_int = 10 .^(level_ip / 10.)

    # Compute integration bounds            
    D = 10 * log10(sum(f_int)) - level_max - 10 * log10(10. / dt)

    # Compute ilevel
    ilevel = level_max + D

    return ilevel
end