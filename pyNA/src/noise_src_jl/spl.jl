function f_spl!(spl, x)

    # x = [c_0, rho_0]
    # y = [spl]
    
    # Compute SPL
    # Remove all SPL below 0
    spl .= 10*log10.(spl) .+ 20*log10(x[1]^2*x[2])
    
    for i in 1:1:size(spl)[1]
        if spl[i] < 1e-99
            spl[i] = 1e-99
        end
    end

end

