using ReverseDiff

function cutoff_spl(spl::Union{Float64, ReverseDiff.TrackedReal})

    if spl < 1e-99
        return 1e-99
    else
        return spl
    end

end

function f_spl!(spl::Array, x::Union{Array, ReverseDiff.TrackedArray})

    # x = [c_0, rho_0]
    # y = [spl]
    
    # Compute SPL
    spl .= 10*log10.(spl) .+ 20*log10(x[1]^2*x[2])
    
    # Remove all SPL below 0
    spl .= cutoff_spl.(spl)

end
