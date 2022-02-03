function f_spl(settings, msap_prop, rho_0, c_0)

    # Compute SPL
    spl = 10*log10.(msap_prop) .+ 20*log10.(rho_0 .* c_0.^2) #.* ones(eltype(msap_prop), (1, settings.N_f))

    # Remove all SPL below 0
    spl = clamp.(spl, 0, Inf)

    return spl
end