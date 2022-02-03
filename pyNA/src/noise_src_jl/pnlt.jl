# Perceived noise level, tone-corrected  
function f_noy(settings, n_t, spl)
    
    # Get type of input vector
    T = eltype(spl)

    # Compute noy
    spl_a = [91. , 85.9, 87.3, 79. , 79.8, 76. , 74. , 74.9, 94.6,  1e8,  1e8, 1e8,  1e8,  1e8,  1e8,  1e8,  1e8,  1e8,  1e8,  1e8,  1e8,  1e8, 44.3, 50.7]
    spl_b = [64, 60, 56, 53, 51, 48, 46, 44, 42, 40, 40, 40, 40, 40, 38, 34, 32, 30, 29, 29, 30, 31, 34, 37]
    spl_c = [52, 51, 49, 47, 46, 45, 43, 42, 41, 40, 40, 40, 40, 40, 38, 34, 32, 30, 29, 29, 30, 31, 34, 37]
    spl_d = [49, 44, 39, 34, 30, 27, 24, 21, 18, 16, 16, 16, 16, 16, 15, 12,  9, 5,  4,  5,  6, 10, 17, 21]
    spl_e = [55, 51, 46, 42, 39, 36, 33, 30, 27, 25, 25, 25, 25, 25, 23, 21, 18, 15, 14, 14, 15, 17, 23, 29]
    m_b   = [0.043478, 0.04057 , 0.036831, 0.036831, 0.035336, 0.033333, 0.033333, 0.032051, 0.030675, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.02996 , 0.02996 , 0.02996 , 0.02996 , 0.02996 , 0.02996 , 0.02996 , 0.042285, 0.042285]
    m_c   = [0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103,      1e8,      1e8,      1e8,      1e8,      1e8,      1e8,      1e8,      1e8,      1e8,      1e8,      1e8,      1e8,      1e8, 0.02996 , 0.02996 ]
    m_d   = [0.07952 , 0.06816 , 0.06816 , 0.05964 , 0.053013, 0.053013, 0.053013, 0.053013, 0.053013, 0.053013, 0.053013, 0.053013, 0.053013, 0.053013, 0.05964 , 0.053013, 0.053013, 0.047712, 0.047712, 0.053013, 0.053013, 0.06816 , 0.07952 , 0.05964 ]
    m_e   = [0.058098, 0.058098, 0.052288, 0.047534, 0.043573, 0.043573, 0.040221, 0.037349, 0.034859, 0.034859, 0.034859, 0.034859, 0.034859, 0.034859, 0.034859, 0.040221, 0.037349, 0.034859, 0.034859, 0.034859, 0.034859, 0.037349, 0.037349, 0.043573]
        
    # Generate function N(SPL) that is always 0 for any SPL
    #N = spl
    #N[findall( spl .< spl_d)] = zeros(T, (n_t, settings.N_f))[findall( spl .< spl_d)]
    ##N[findall(spl_a .<= spl          )] =       10 .^(m_c .* (spl - spl_c))[findall(spl_a .<= spl          )]
    #N[findall(spl_b .<= spl .<= spl_a)] =       10 .^(m_b .* (spl - spl_b))[findall(spl_b .<= spl .<= spl_a)]
    #N[findall(spl_e .<= spl .<= spl_b)] = 0.3 * 10 .^(m_e .* (spl - spl_e))[findall(spl_e .<= spl .<= spl_b)]
    #N[findall(spl_d .<= spl .<= spl_e)] = 0.1 * 10 .^(m_d .* (spl - spl_d))[findall(spl_d .<= spl .<= spl_e)]

    #N = spl.^ .- 1

    N = zeros(T, (n_t, settings.N_f))
    for i in range(1, n_t, step=1)
        for j in range(1, settings.N_f, step=1)
            if spl_a[j] <= spl[i,j]
                N[i,j] = 10^(m_c[j] * (spl[i,j] - spl_c[j]))
            elseif spl_b[j] <= spl[i,j] <= spl_a[j]
                N[i,j] = 10^(m_b[j] * (spl[i,j] - spl_b[j]))
            elseif spl_e[j] <= spl[i,j] <= spl_b[j]
                N[i,j] = 0.3 * 10^(m_e[j] * (spl[i,j] - spl_e[j]))
            elseif spl_d[j] <= spl[i,j] <= spl_e[j]
                N[i,j] = 0.1 * 10^(m_d[j] * (spl[i,j] - spl_d[j]))
            else
                N[i,j] = 0.
            end
        end
    end

    #N[findall(spl_a.* ones(T, (n_t, 1)) .<= spl)] = 10 .^(m_c.* ones(T, (n_t,1)) .* (spl - spl_c.* ones(T, (n_t,1))))[findall(spl_a.* ones(T, (n_t,1)) .<= spl)]
    #N[findall(spl_b.* ones(T, (n_t, 1)) .<= spl .<= spl_a.* ones(T, (n_t,1)) )] = 10 .^(m_b.* ones(T, (n_t,1)) .* (spl - spl_b.* ones(T, (n_t,1))))[findall(spl_b.* ones(T, (n_t,1)) .<= spl .<= spl_a.* ones(T, (n_t,1)) )]
    #N[findall(spl_e.* ones(T, (n_t, 1)) .<= spl .<= spl_b.* ones(T, (n_t,1)) )] = 0.3 * 10 .^(m_e.* ones(T, (n_t,1)) .* (spl - spl_e.* ones(T, (n_t,1))))[findall(spl_e.* ones(T, (n_t,1)) .<= spl .<= spl_b.* ones(T, (n_t,1)) )]
    #N[findall(spl_d.* ones(T, (n_t, 1)) .<= spl .<= spl_e.* ones(T, (n_t,1)) )] = 0.1 * 10 .^(m_d.* ones(T, (n_t,1)) .* (spl - spl_d.* ones(T, (n_t,1))))[findall(spl_d.* ones(T, (n_t,1)) .<= spl .<= spl_e.* ones(T, (n_t,1)) )]
    
    return N
end

function f_pnl(n_t, N)

    pnl = zeros(eltype(N), n_t)

    n_max = maximum(N, dims=2)

    n = n_max .+ 0.15*(sum(N, dims=2) .- n_max)
    
    pnl = 40. .+ 10. / log10(2) * log10.(n)

    # Remove all PNL below 0
    pnl = clamp.(pnl, 0, Inf)
    
    return pnl
end

function f_tone_corrections(settings, n_t, spl)

    # Get type of input vector
    T = eltype(spl)

    # Step 1: Compute the slope of SPL
    s = zeros(T, (n_t, settings.N_f))    
    s[:,4:end] = @view(spl[:,4:end]) .- @view(spl[:,3:end-1])
    
    # Step 2: Compute the absolute value of the slope and compare to 5    
    slope = zeros(T, (n_t, settings.N_f))    
    slope[:,4:end] = @view(s[:,4:end]) .- @view(s[:,3:end-1])

    slope_large = zeros(T, (n_t, settings.N_f))  
    idx_slope_large = findall(abs.(slope).>5)
    slope_large[idx_slope_large] = ones(size(spl))[idx_slope_large]

    # Step 3: Compute the encircled values of SPL
    spl_large = zeros(T, (n_t, settings.N_f))   
    spl_large[findall((slope_large .== 1) .* (s.> 0) .* (s .- hcat(zeros(T, (n_t, )),s[:,1:end-1]) .> 0) )] = ones(size(spl))[findall( (slope_large .== 1) .* (s.>0) .* (s .- hcat(zeros(T, (n_t, )),s[:,1:end-1]) .> 0) )]
    interm = hcat(zeros(T, (n_t,1)), spl_large)
    interm[findall( (slope_large .== 1) .* (s.<=0) .* (hcat(zeros(T, (n_t, )),s[:,1:end-1]) .> 0) )] = ones(n_t, settings.N_f)[findall( (slope_large .== 1) .* (s.<=0) .* (hcat(zeros(T, (n_t, )),s[:,1:end-1]) .> 0) )]
    spl_large = interm[:,2:end]    
        
    # Step 4: Compute new adjusted sound pressure levels SPL'
    spl_p = zeros(T, (n_t, settings.N_f))    
    spl_p[:,24] = spl[:,22] + s[:,22]
    spl_p[:,2:23]= 0.5 * (spl[:,1:22] .+ spl[:,3:24])
    spl_p[findall(spl_large .==0)] = spl[findall(spl_large .==0)]
    
    # Step 5: Recompute the slope s'
    s_p = zeros(T, (n_t, settings.N_f + 1))
    s_p[:,4:end-1] = @view(spl_p[:,4:end]) .- @view(spl_p[:,3:end-1])
    s_p[:,3] = @view(s_p[:,4])
    # Compute 25th imaginary band
    s_p[:,25] = @view(s_p[:,24])

    # Step 6: Compute arithmetic average of the 3 adjacent slopes
    s_bar = zeros(T, (n_t, settings.N_f))
    s_bar[:,3:end-1] = 1. / 3. * (@view(s_p[:,3:end-2]) + @view(s_p[:,4:end-1]) + @view(s_p[:,5:end]))

    # Step 7: Compute final 1/3 octave-band sound pressure level       
    spl_pp = zeros(T, (n_t, settings.N_f))
    spl_pp[:, 3] = spl[:,3]
    spl_pp[:, 4:end]  = cumsum(s_bar[:, 3:end-1], dims=2) .+  spl[:,3]
       
    # Step 8: Compute the difference between SPL and SPL_pp
    F = zeros(T, (n_t, settings.N_f))
    F[3:end] = spl[3:end] .- spl_pp[3:end]
    F[findall(F.<1.5)] = zeros(T, (n_t, settings.N_f))[findall(F.<1.5)]
    
    # Step 9: Compute the correction factor C    
    c0  = zeros(T, (n_t, 2))
    c10 = zeros(T, (n_t, 8))
    c10[findall(1.5 .<= F[:,3:10] .< 3 )]    = @view(F[:,3:10][findall(1.5 .<= F[:,3:10] .< 3 )]) / 3. .- 0.5
    c10[findall(3.0 .<= F[:,3:10] .< 20)]    = @view(F[:,3:10][findall(3.0 .<= F[:,3:10] .< 20)]) / 6.
    c10[findall(20. .<= F[:,3:10]      )]    = @view(F[:,3:10][findall(20. .<= F[:,3:10]      )]) * 0. .+ (3 .+ 1/3.)
    c20 = zeros(T, (n_t, 10))
    c20[findall(1.5 .<= F[:,11:20] .< 3 )] = @view(F[:,11:20][findall(1.5 .<= F[:,11:20] .< 3 )])*2 / 3. .- 1.   
    c20[findall(3.0 .<= F[:,11:20] .< 20)] = @view(F[:,11:20][findall(3.0 .<= F[:,11:20] .< 20)])/3.   
    c20[findall(20. .<= F[:,11:20]      )] = @view(F[:,11:20][findall(20. .<= F[:,11:20]      )])*0. .+ (6 .+ 2/3.)
    cend = zeros(T, (n_t, 4))
    cend[findall(1.5 .<= F[:,21:end] .< 3 )] = @view(F[:,21:end][findall(1.5 .<= F[:,21:end] .< 3 )])/3. .- 0.5 
    cend[findall(3.0 .<= F[:,21:end] .< 20)] = @view(F[:,21:end][findall(3.0 .<= F[:,21:end] .< 20)])/6.  
    cend[findall(20. .<= F[:,21:end]      )] = @view(F[:,21:end][findall(20. .<= F[:,21:end]      )])*0. .+ (3. .+ 1/3.)  
    C = hcat(c0, c10, c20, cend)
  
    return C
end

function f_pnlt(settings, data, n_t, spl)

    # Compute noy
    T = eltype(spl)
    func_noy = LinearInterpolation((data.noy_spl, data.noy_freq), data.noy)
    N = func_noy.(spl, reshape(data.f, (1,settings.N_f)).*ones(T, (n_t,1)))
    # N = f_noy(settings, n_t, spl)

    # Compute pnl
    pnl = f_pnl(n_t, N)
    
    # Compute the correction factor C
    C = f_tone_corrections(settings, n_t, spl)
    
    # Step 10: Compute the largest of the tone correction
    if settings.TCF800
        c_max = maximum(C[:,14:end], dims=2)
    else
        c_max = maximum(C, dims=2)
    end

    # Compute PNLT    
    pnlt = pnl + c_max

    return pnlt, C
end