function f_pnl(N)

    # Compute perceived noise level
    n_max = maximum(N)
    n = n_max + 0.15*(sum(N) - n_max)
    pnl = 40 + 10 / log10(2) * log10(n)
    
    return pnl

end

function f_pnl!(pnl, N)

    # Compute perceived noise level
    n_max = maximum(N)
    n = n_max + 0.15*(sum(N) - n_max)
    pnl .= 40 + 10 / log10(2) * log10(n)
    
end