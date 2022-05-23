# Effective perceived noise level
function f_epnl(t_o, pnlt)

    # Number of time steps
    n_t = size(t_o)[1]

    # Interpolate time, pnlt and C
    dt = 0.5
    n_ip = Int64(ceil((t_o[end]-t_o[1])/dt))

    T = eltype(pnlt)
    t_ip = zeros(T, n_ip)
    for i in range(1, n_ip, step=1)
        t_ip[i] = t_o[1] + (i-1)*dt
    end

    # Interpolate the data
    #f_ipnlterp = PCHIPInterpolation.Interpolator(t_o, reshape(pnlt, (n_t, )))
    f_ipnlterp = LinearInterpolation(t_o, reshape(pnlt, (n_t, )))
    pnlt_ip = f_ipnlterp.(t_ip)
        
    # Compute max. PNLT
    pnltm = maximum(pnlt_ip)

    # Compute max. PNLT point (k_m)
    I = findall(x->x>pnltm - 10.,pnlt_ip)

    # ICAO Annex 16 procedures (p132 - App. 2-18)
    f_int = 10 .^(pnlt_ip / 10.)

    # Compute duration correction
    if pnltm > 10
        # Compute integration bounds: lower bound
        i_1 = I[1]
        if i_1 != 1 
            if abs(pnlt_ip[i_1] - (pnltm - 10)) > abs(pnlt_ip[i_1 - 1] - (pnltm - 10))
                i_1 = i_1 - 1
            end
        end

        # Compute integration bounds: upper bound
        i_2 = I[end]
        if i_2 != size(pnlt_ip)[1]
            if (abs(pnlt_ip[i_2] - (pnltm - 10))) > (abs(pnlt_ip[i_2 + 1] - (pnltm - 10)))
                i_2 = i_2 + 1
            end
        end
         
        # Duration correction
        D = 10 * log10(sum(f_int[i_1:i_2])) - pnltm - 10 * log10(10. / dt)
    else
        # Duration correction
        D = 10 * log10(sum(f_int)) - pnltm - 10 * log10(10. / dt)
    end

    # Compute EPNL
    epnl = pnltm + D

    return epnl
end