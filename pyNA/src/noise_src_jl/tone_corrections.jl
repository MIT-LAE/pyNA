function f_tone_corrections(settings, spl)

    # Get type of input vector
    T = eltype(spl)

    # Step 1: Compute the slope of SPL
    s = zeros(T, (settings["n_frequency_bands"], ))    
    s[4:end] = spl[4:end] .- spl[3:end-1]
    
    # Step 2: Compute the absolute value of the slope and compare to 5    
    slope = zeros(T, (settings["n_frequency_bands"], ))    
    slope[4:end] = s[4:end] .- s[3:end-1]

    slope_large = zeros(T, (settings["n_frequency_bands"], ))  
    idx_step2 = findall(abs.(slope).>5)
    slope_large[idx_step2] = ones(T, (settings["n_frequency_bands"], ))[idx_step2]

    # Step 3: Compute the encircled values of SPL
    spl_large = zeros(T, (settings["n_frequency_bands"], ))   
    idx_step3a = findall((slope_large .== 1) .* (s.> 0) .* (s .- vcat(0, s[1:end-1]) .> 0) )
    spl_large[idx_step3a] = ones(T, (settings["n_frequency_bands"],))[idx_step3a]
    
    interm = vcat(0, spl_large)
    idx_step3b = findall( (slope_large .== 1) .* (s.<=0) .* (vcat(0, s[1:end-1]) .> 0) )
    interm[idx_step3b] = ones(settings["n_frequency_bands"], )[idx_step3b]
    spl_large = interm[2:end]

    # Step 4: Compute new adjusted sound pressure levels SPL'
    spl_p = zeros(T, (settings["n_frequency_bands"], ))    
    spl_p[24] = spl[22] + s[22]
    spl_p[2:23]= 0.5 * (spl[1:22] .+ spl[3:24])

    idx_step4 = findall(spl_large .==0)
    spl_p[idx_step4] = spl[idx_step4]

    # Step 5: Recompute the slope s'
    s_p = zeros(T, (settings["n_frequency_bands"] + 1,))
    s_p[4:end-1] = spl_p[4:end] .- spl_p[3:end-1]
    s_p[3] = s_p[4]
    # Compute 25th imaginary band
    s_p[25] = s_p[24]

    # Step 6: Compute arithmetic average of the 3 adjacent slopes
    s_bar = zeros(T, (settings["n_frequency_bands"],))
    s_bar[3:end-1] = 1/3 * (s_p[3:end-2] + s_p[4:end-1] + s_p[5:end])

    # Step 7: Compute final 1/3 octave-band sound pressure level       
    spl_pp = zeros(T, (settings["n_frequency_bands"],))
    spl_pp[3] = spl[3]
    spl_pp[4:end]  = cumsum(s_bar[3:end-1]) .+  spl[3]
       
    # Step 8: Compute the difference between SPL and SPL_pp
    F = zeros(T, (settings["n_frequency_bands"],))
    F[3:end] = spl[3:end] .- spl_pp[3:end]
    
    idx_step8 = findall(F.<1.5)
    F[idx_step8] = zeros(T, (settings["n_frequency_bands"],))[idx_step8]
    
    # Step 9: Compute the correction factor C    
    c10 = zeros(T, (8,))
    idx_step9a1 = findall(1.5 .<= F[3:10] .< 3 )
    idx_step9a2 = findall(3.0 .<= F[3:10] .< 20)
    idx_step9a3 = findall(20. .<= F[3:10]      )
    c10[idx_step9a1] = F[3:10][idx_step9a1] / 3. .- 0.5
    c10[idx_step9a2] = F[3:10][idx_step9a2] / 6.
    c10[idx_step9a3] = F[3:10][idx_step9a3] * 0. .+ (3 .+ 1/3.)

    c20 = zeros(T, (10,))
    idx_step9b1 = findall(1.5 .<= F[11:20] .< 3 )
    idx_step9b2 = findall(3.0 .<= F[11:20] .< 20)
    idx_step9b3 = findall(20. .<= F[11:20]      )
    c20[idx_step9b1] = F[11:20][idx_step9b1] * 2 / 3. .- 1.   
    c20[idx_step9b2] = F[11:20][idx_step9b2] / 3.   
    c20[idx_step9b3] = F[11:20][idx_step9b3] * 0. .+ (6 .+ 2/3.)

    cend = zeros(T, (4,))
    idx_step9c1 = findall(1.5 .<= F[21:end] .< 3 )
    idx_step9c2 = findall(3.0 .<= F[21:end] .< 20)
    idx_step9c3 = findall(20. .<= F[21:end]      )
    cend[idx_step9c1] = F[21:end][idx_step9c1] / 3. .- 0.5 
    cend[idx_step9c2] = F[21:end][idx_step9c2] / 6.  
    cend[idx_step9c3] = F[21:end][idx_step9c3] * 0. .+ (3. .+ 1/3.)  
    C = vcat(0, 0, c10, c20, cend)
      
    # Compute the largest of the tone correction
    if settings["tones_under_800Hz"]
        c_max = maximum(C)
    else
        c_max = maximum(C[14:end])
    end

    return c_max
end