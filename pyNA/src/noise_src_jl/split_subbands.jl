function split_subbands!(spl_sb, spl, settings)
    
    # x = spl
    # y = spl_sb
    
    # Integer for subband calculation
    m = Int64((settings["n_frequency_subbands"] - 1) / 2)
 
    # Calculate slope of spectrum [1, N_f]
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 8-9
    for i in 1:1:settings["n_frequency_bands"]
                
        if i == 1
            u = spl[2]/spl[1]
            v = spl[2]/spl[1]
        elseif i == settings["n_frequency_bands"]
            u = spl[end]/spl[end - 1]
            v = spl[end]/spl[end - 1]
        else
            u = spl[i]/spl[i-1] 
            v = spl[i+1]/spl[i]
        end
        
        A = 1
        for h in 1:m
            A += u^((h-m-1)/settings["n_frequency_subbands"]) + v^(h/settings["n_frequency_subbands"])
        end
        
        spl_sb[(i-1)*settings["n_frequency_subbands"] .+ (1:2)] = spl[i]/A .* u.^ (((1:2) .- m .-1) / settings["n_frequency_subbands"])
        spl_sb[(i-1)*settings["n_frequency_subbands"] .+ 3]     = spl[i]/A
        spl_sb[(i-1)*settings["n_frequency_subbands"] .+ (4:5)] = spl[i]/A .* v.^ (((4:5) .- m .-1) / settings["n_frequency_subbands"])
        
    end
    
end

split_subbands_fwd! = (y, x)->split_subbands!(y, x, settings)