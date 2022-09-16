using PCHIPInterpolation

function f_ilevel(x, settings)

    # x = [t_o, level]
    # y = ilevel
    
    # Number of time steps
    n_t = Int64(size(x)[1]/2)

    # Interpolate time, level
    n_ip = Int64(ceil((x[n_t]-x[1])/settings["epnl_dt"]))

    t_ip = zeros(eltype(x), n_ip)
    for i in range(1, n_ip, step=1)
        t_ip[i] = x[1] + (i-1)*settings["epnl_dt"]
    end
    
    # Interpolate the data
    f_ilevel_interp = PCHIPInterpolation.Interpolator(x[1:n_t], x[n_t+1:end])
    level_ip = f_ilevel_interp.(t_ip)
        
    # Compute max. level
    level_max = maximum(level_ip)

    # ICAO Annex 16 procedures (p132 - App. 2-18)
    f_int = 10 .^(level_ip / 10.)

    # Compute integration bounds            
    D = 10 * log10(sum(f_int)) - level_max - 10 * log10(10. / settings["epnl_dt"])

    # Compute ilevel
    ilevel = level_max + D

    return ilevel
end

function f_ilevel!(ilevel, x, settings)

    # x = [t_o, level]
    # y = ilevel
    
    # Number of time steps
    n_t = Int64(size(x)[1]/2)

    # Interpolate time, level
    n_ip = Int64(ceil((x[n_t]-x[1])/settings["epnl_dt"]))

    t_ip = zeros(eltype(x), n_ip)
    for i in range(1, n_ip, step=1)
        t_ip[i] = x[1] + (i-1)*settings["epnl_dt"]
    end
    
    # Interpolate the data
    f_ilevel_interp = PCHIPInterpolation.Interpolator(x[1:n_t], x[n_t+1:end])
    level_ip = f_ilevel_interp.(t_ip)
        
    # Compute max. level
    level_max = maximum(level_ip)

    # ICAO Annex 16 procedures (p132 - App. 2-18)
    f_int = 10 .^(level_ip / 10.)

    # Compute integration bounds            
    D = 10 * log10(sum(f_int)) - level_max - 10 * log10(10. / settings["epnl_dt"])

    # Compute ilevel
    ilevel .= level_max + D
end

f_ilevel_fwd! = (y,x) -> f_ilevel!(y, x, settings)