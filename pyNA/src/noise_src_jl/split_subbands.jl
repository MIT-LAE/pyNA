function split_subbands(settings, spl)
    
    T = eltype(spl)

    # Integer for subband calculation
    m = Int64((settings["n_frequency_subbands"] - 1) / 2)
 
    # Calculate slope of spectrum
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 8-9
    u = zeros(T, (1, settings["n_frequency_bands"]))
    v = zeros(T, (1, settings["n_frequency_bands"]))
    u[2:end]   = spl[2:end] ./ spl[1:end-1] 
    v[2:end-1] = spl[3:end] ./ spl[2:end-1]
    u[1] = v[1] = spl[2] ./ spl[1]
    u[end] = v[end] = spl[end] ./ spl[end - 1]

    # Calculate constant A
    h = reshape(1:1:m, (m, 1))
    A = sum((u.^((h .- m .- 1)/settings["n_frequency_subbands"]) .+ v.^(h/settings["n_frequency_subbands"])), dims=1)
    A = A + ones(T, (1, settings["n_frequency_bands"]))

    h = 1:1:settings["n_frequency_subbands"]
    
    spl_sb = zeros(T, (settings["n_frequency_subbands"], settings["n_frequency_bands"], ))
    spl_sb[1:m,:] = ((reshape(spl, (1, settings["n_frequency_bands"]))./A) .* u .^ ((h .- m .-1) / settings["n_frequency_subbands"]))[1:m,:]
    spl_sb[m+1,:] = (reshape(spl, (1, settings["n_frequency_bands"]))./A)
    spl_sb[m+2:end,:] = ((reshape(spl, (1, settings["n_frequency_bands"]))./A) .* v .^ ((h .- m .-1) / settings["n_frequency_subbands"]))[m+2:end,:]
    spl_sb = reshape(spl_sb, (settings["n_frequency_bands"]*settings["n_frequency_subbands"], ))
    
    return spl_sb
end