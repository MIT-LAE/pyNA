using Interpolations

# A-weighted sound pressure level
function f_aspl(data, spl)
    
    f_AW = LinearInterpolation(data.aw_freq, data.aw_db)

    # Number of time steps
    n_t = size(spl)[1]

    aspl = zeros(eltype(spl), n_t)
    for i in 1:1:n_t        
        weights = f_AW(data.f)

        aspl[i] = 10 * log10.( sum(10 .^((spl[i,:] .+ weights)./10.) ) )
    end

    return aspl
end