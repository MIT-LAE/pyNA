function interpolate_level_timeseries(t_o, level, dt)

    n_t = size(t_o)[1]
    n_ip = Int64(ceil((t_o[end]-t_o[1])/dt))

    T = eltype(level)
    t_interpolated = zeros(T, n_ip)
    for i in range(1, n_ip, step=1)
        t_interpolated[i] = t_o[1] + (i-1)*dt
    end

    # Interpolate the data
    f_level_interp = PCHIPInterpolation.Interpolator(t_o, reshape(level, (n_t, )))
    level_interpolated = f_level_interp.(t_interpolated)

    return level_interpolated
end