function f_sel(t_o, aspl)

    # Number of time steps
    n_t = size(t_o)[1]

    # Interpolate time, aspl
    dt = 0.5
    n_ip = Int64(ceil((t_o[end]-t_o[1])/dt))

    T = eltype(aspl)
    t_ip = zeros(T, n_ip)
    for i in range(1, n_ip, step=1)
        t_ip[i] = t_o[1] + (i-1)*dt
    end

    # Interpolate the data
    f_aspl_interp = PCHIPInterpolation.Interpolator(t_o, reshape(aspl, (n_t, )))
    aspl_ip = f_aspl_interp.(t_ip)
        
    # Compute max. PNLT
    asplm = maximum(aspl_ip)

    # ICAO Annex 16 procedures (p132 - App. 2-18)
    f_int = 10 .^(aspl_ip / 10.)

    # Compute integration bounds            
    D = 10 * log10(sum(f_int)) - asplm - 10 * log10(10. / dt)

    # Compute sel
    sel = asplm + D

    return sel
end