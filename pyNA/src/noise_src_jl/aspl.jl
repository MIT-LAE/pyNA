function f_aspl(spl, pyna_ip, f)
    
    # Get a-weights
    weights = pyna_ip.f_aw(f)
    
    aspl = 10*log10(sum(10 .^((spl .+ weights)./10.)))

    return aspl

end

function f_aspl!(aspl, spl, pyna_ip, f)
    
    # Get a-weights
    weights = pyna_ip.f_aw(f)
    
    aspl .= 10*log10.(sum(10 .^((spl .+ weights)./10.)))

end

f_aspl_fwd! = (y,x)-> f_aspl!(y, x, pyna_ip, f)