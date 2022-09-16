function f_tone_corrections!(C, spl, settings)

    # Step 1: Compute the slope of SPL
    s = zeros(eltype(spl), (settings["n_frequency_bands"], ))    
    s[4:end] = spl[4:end] .- spl[3:end-1]
    
    # Step 2: Compute the absolute value of the slope and compare to 5
    # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 2
    slope = zeros(eltype(spl), settings["n_frequency_bands"])
    slope_large = zeros(eltype(spl), settings["n_frequency_bands"])
    
    for i in 4:1:settings["n_frequency_bands"]
        # Compute the absolute value of the slope
        slope[i] = s[i] - s[i - 1]
        # Check if slope is larger than 5
        if abs(slope[i]) > 5
            slope_large[i] = 1
        end
    end

    # Step 3: Compute the encircled values of SPL
    spl_large = zeros(eltype(spl), settings["n_frequency_bands"])        
    for j in 1:1:settings["n_frequency_bands"]
        # Check if value of slope is encircled
        if slope_large[j] == 1
            # Check if value of slope is positive and greater than previous slope
            if (s[j] > 0) & (s[j] > s[j - 1])
                spl_large[j] = 1
            elseif (s[j] <= 0) & (s[j - 1] > 0)
                spl_large[j - 1] = 1
            end
        end
    end

    # Step 4: Compute new adjusted sound pressure levels SPL"
    spl_p = zeros(eltype(spl), settings["n_frequency_bands"])      
    for j in 1:1:settings["n_frequency_bands"]
        if spl_large[j] == 0
            spl_p[j] = spl[j]
        elseif spl_large[j] == 1
            if j <= 23
                spl_p[j] = 0.5 * (spl[j - 1] + spl[j + 1])
            elseif j == 24
                spl_p[j] = spl[22] + s[22]
            end
        end
    end

    # Step 5: Recompute the slope s'
    s_p = zeros(eltype(spl), (settings["n_frequency_bands"] + 1,))
    s_p[4:end-1] = spl_p[4:end] .- spl_p[3:end-1]
    s_p[3] = s_p[4]
    # Compute 25th imaginary band
    s_p[25] = s_p[24]

    # Step 6: Compute arithmetic average of the 3 adjacent slopes
    s_bar = zeros(eltype(spl), (settings["n_frequency_bands"],))
    s_bar[3:end-1] = 1/3 * (s_p[3:end-2] + s_p[4:end-1] + s_p[5:end])

    # Step 7: Compute final 1/3 octave-band sound pressure level
    spl_pp = zeros(eltype(spl), settings["n_frequency_bands"])        
    for j in range(2, settings["n_frequency_bands"], step=1)
        if j == 3
            spl_pp[j] = spl[j]
        elseif j > 3
            spl_pp[j] = spl_pp[j - 1] + s_bar[j - 1]
        end
    end
       
    # Step 8: Compute the difference between SPL and SPL_pp
    F = zeros(eltype(spl), (settings["n_frequency_bands"],))
    F[3:end] = spl[3:end] .- spl_pp[3:end]
    for i in 1:1:settings["n_frequency_bands"]
        # Check values larger than 1.5 (ICAO Appendix 2-16)
        if F[i] < 1.5
            F[i] = 0
        end
    end
    
    # Step 9: Compute the correction factor C    
    for j in 1:1:settings["n_frequency_bands"]
        if j < 11  # Frequency in [50,500[
            if 1.5 <= F[j] < 3
                C[j] = F[j] / 3. - 0.5
            elseif 3. <= F[j] < 20.
                C[j] = F[j] / 6.
            elseif F[j] >= 20.
                C[j] = 3. + 1 / 3.
            end

        elseif 10 <= j <= 20  # Frequency in [500,5000]
            if 1.5 <= F[j] < 3.
                C[j] = 2. * F[j] / 3. - 1.0
            elseif 3 <= F[j] < 20
                C[j] = F[j] / 3.
            elseif F[j] >= 20
                C[j] = 6. + 2. / 3.
            end

        elseif j > 20  # Frequency in ]5000,10000]
            if 1.5 <= F[j] < 3.
                C[j] = F[j] / 3. - 0.5
            elseif 3. <= F[j] < 20.
                C[j] = F[j] / 6.
            elseif F[j] >= 20.
                C[j] = 3. + 1. / 3.
            end
        end
    end

end

f_tone_corrections_fwd! = (y,x)->f_tone_corrections!(y, x, settings)