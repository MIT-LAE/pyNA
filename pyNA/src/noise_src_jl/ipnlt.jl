function f_ipnlt(settings, n_t, pnlt, t_o)

    # Interpolate time, pnlt and C
    dt = 0.5
    n_ip = Int64(ceil((t_o[end]-t_o[1])/dt))

    T = eltype(pnlt)
    t_ip = zeros(T, n_ip)
    for i in range(1, n_ip, step=1)
        t_ip[i] = t_o[1] + (i-1)*dt
    end

    # Interpolate the data
    f_ipnlterp = PCHIPInterpolation.Interpolator(t_o, reshape(pnlt, (n_t, )))
    pnlt_ip = f_ipnlterp.(t_ip)
        
    # Compute max. PNLT
    pnltm = maximum(pnlt_ip)

    # ICAO Annex 16 procedures (p132 - App. 2-18)
    f_int = 10 .^(pnlt_ip / 10.)

    # Compute integration bounds            
    D = 10 * log10(sum(f_int)) - pnltm - 10 * log10(10. / dt)

    # Compute ipnlt
    ipnlt = pnltm + D

    return ipnlt
end