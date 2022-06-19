function f_pnl(N)

    # Compute perceived noise level
    n_max = maximum(N)
    n = n_max + 0.15*(sum(N) - n_max)
    pnl = 40 + 10 / log10(2) * log10(n)

    # Remove all PNL below 0
    pnl = clamp(pnl, 0, Inf)
    
    return pnl
end