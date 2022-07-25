function f_aspl(pyna_ip, f, spl)
    
    # Get a-weights
    weights = pyna_ip.f_aw(f)
    
    aspl = 10*log10.(sum(10 .^((spl .+ weights)./10.)))
    
    return aspl
end