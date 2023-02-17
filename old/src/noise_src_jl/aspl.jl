using ReverseDiff

function f_aspl(spl::Union{Array, ReverseDiff.TrackedArray}, pyna_ip, f::Array{Float64, 1})
    
    # Get a-weights
    weights = pyna_ip.f_aw(f)
    
    return 10*log10(sum(10 .^((spl .+ weights)./10.)))

end

f_aspl_fwd = (x) -> f_aspl(x, pyna_ip, f)

function f_aspl!(aspl::Array, spl::Union{Array, ReverseDiff.TrackedArray}, pyna_ip, f::Array{Float64, 1})
    
    # Get a-weights
    weights = pyna_ip.f_aw(f)
    
    aspl .= 10*log10.(sum(10 .^((spl .+ weights)./10.)))

end

f_aspl_fwd! = (y,x)-> f_aspl!(y, x, pyna_ip, f)