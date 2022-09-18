using ReverseDiff

function f_oaspl(x::Union{Array, ReverseDiff.TrackedArray})
    
    # y = oaspl
    # x = spl
    
    # Compute 
    oaspl = 10*log10(sum(10 .^(x./10)))

    return oaspl
    
end


function f_oaspl!(oaspl::Array, x::Union{Array, ReverseDiff.TrackedArray})
    
    # y = oaspl
    # x = spl
    
    # Compute 
    oaspl .= 10*log10(sum(10 .^(x./10)))
    
end
