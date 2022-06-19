function f_oaspl(spl)
    
    # Compute 
    oaspl = 10*log10(sum(10 .^(spl./10)))

    return oaspl
end