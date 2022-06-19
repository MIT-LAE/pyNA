function f_spl!(spl, rho_0, c_0)

    # Compute SPL
    @. spl = 10*log10.(spl) + 20*log10(rho_0 * c_0^2)

    # Remove all SPL below 0
    @. spl = clamp(spl, 0, Inf)

end

