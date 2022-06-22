function f_aspl(f, spl)
    
    # Get a-weights
    weights = f_aw(f)
    
    aspl = 10*log10.(sum(10 .^((spl .+ weights)./10.)))
    
    return aspl
end