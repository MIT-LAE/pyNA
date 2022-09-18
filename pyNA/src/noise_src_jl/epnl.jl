using PCHIPInterpolation

function f_epnl(x::Union{Array, ReverseDiff.TrackedArray}, settings)

    # x = [t_o, level]
    # y = epnl
    
    # Number of time steps
    n_t = Int64(size(x)[1]/2)

    # Interpolate time, pnlt and C
    n_ip = Int64(ceil((x[n_t]-x[1])/settings["epnl_dt"]))
    
    t_ip = zeros(eltype(x), (n_ip,))
    for i in range(1, n_ip, step=1)
        t_ip[i] = x[1] + (i-1)*settings["epnl_dt"]
    end

    # Interpolate the data
    f_ipnlt_interp = PCHIPInterpolation.Interpolator(x[1:n_t], x[n_t+1:end])
    pnlt_ip = f_ipnlt_interp.(t_ip)

    # Compute max. PNLT
    pnltm = maximum(pnlt_ip)

    # Check tone band-sharing
    # I_max = findall(x->x==pnltm, pnlt_ip)
    # if settings["epnl_bandshare"]
    #     continue
    # end
    
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
        D = 10 * log10(sum(f_int[i_1:i_2])) - pnltm - 10 * log10(10. / settings["epnl_dt"])
    else
        # Duration correction
        D = 10 * log10(sum(f_int)) - pnltm - 10 * log10(10. / settings["epnl_dt"])
    end
    
    # Compute EPNL
    epnl = pnltm + D

    return epnl

end

function f_epnl!(epnl::Array, x::Union{Array, ReverseDiff.TrackedArray}, settings)

    # x = [t_o, level]
    # y = epnl
    
    # Number of time steps
    n_t = Int64(size(x)[1]/2)

    # Interpolate time, pnlt and C
    n_ip = Int64(ceil((x[n_t]-x[1])/settings["epnl_dt"]))
    
    t_ip = zeros(eltype(x), (n_ip,))
    for i in range(1, n_ip, step=1)
        t_ip[i] = x[1] + (i-1)*settings["epnl_dt"]
    end

    # Interpolate the data
    f_ipnlt_interp = PCHIPInterpolation.Interpolator(x[1:n_t], x[n_t+1:end])
    pnlt_ip = f_ipnlt_interp.(t_ip)

    # Compute max. PNLT
    pnltm = maximum(pnlt_ip)

    # Check tone band-sharing
    # I_max = findall(x->x==pnltm, pnlt_ip)
    # if settings["epnl_bandshare"]
    #     continue
    # end
    
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
        D = 10 * log10(sum(f_int[i_1:i_2])) - pnltm - 10 * log10(10. / settings["epnl_dt"])
    else
        # Duration correction
        D = 10 * log10(sum(f_int)) - pnltm - 10 * log10(10. / settings["epnl_dt"])
    end
    
    # Compute EPNL
    epnl .= pnltm + D

end

f_epnl_fwd! = (y,x)->f_epnl!(y, x, settings)