function f_ioaspl(t_o, oaspl)

    # Number of time steps
    n_t = size(t_o)[1]

    # Interpolate time, oaspl and C
    dt = 0.5
    n_ip = Int64(ceil((t_o[end]-t_o[1])/dt))

    T = eltype(oaspl)
    t_ip = zeros(T, n_ip)
    for i in range(1, n_ip, step=1)
        t_ip[i] = t_o[1] + (i-1)*dt
    end

    # Interpolate the data
    f_ioaspl_itp = PCHIPInterpolation.Interpolator(t_o, reshape(oaspl, (n_t, )))
    oaspl_ip = f_ioaspl_itp.(t_ip)
        
    # Compute max. OASPL
    oasplm = maximum(oaspl_ip)

    # ICAO Annex 16 procedures (p132 - App. 2-18)
    f_int = 10 .^(oaspl_ip / 10.)

    # Compute integration bounds            
    D = 10 * log10(sum(f_int)) - oasplm - 10 * log10(10. / dt)

    # Compute ioaspl
    ioaspl = oasplm + D

    return ioaspl
end