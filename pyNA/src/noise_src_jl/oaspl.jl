# Overall sound pressure level
function f_oaspl(settings, spl)
    
    # Compute 
    oaspl = (10 * log10.( sum(10 .^(spl./10.), dims=2) ))

    return oaspl
end