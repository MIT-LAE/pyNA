function inlet_broadband(pyna_ip, settings, theta, M_tip, tsqem, M_d_fan::Float64, RSS_fan::Float64)

    T = eltype(M_tip)
    # Fan inlet broadband noise component:
    if settings["fan_BB_method"] == "original"
        # Tip Mach-dependent term (F1 of Eqn 4 in report, Figure 4A):
        if M_d_fan <= 1
            if M_tip <= 0.9
                F1IB = 58.5
            else
                F1IB = 58.5 - 20 * log10(M_tip / 0.9)
            end
        else
            if M_tip <= 0.9
                F1IB = 58.5 + 20 * log10(M_d_fan)
            else
                F1IB = 58.5 + 20 * log10(M_d_fan) - 20 * log10(M_tip / 0.9)
            end
        end

        # Rotor-stator correction term (F2 of Eqn 4, Figure 6B):
        if settings["fan_id"] == false
            F2IB = -5 * log10(RSS_fan / 300)  # If no distortion
        else
            if RSS_fan <= 100
                F2IB = -5 * log10(RSS_fan / 300)
            else
                F2IB = -5 * log10(100 / 300)  # This is set to a constant 2.3856
            end
        end

    elseif settings["fan_BB_method"] == "allied_signal"
        # Tip Mach-dependent term (F1 of Eqn 4 in report, Figure 4A, modified by AlliedSignal):
        if M_d_fan <= 1
            if M_tip <= 0.9
                F1IB = 55.5
            else
                F1IB = 55.5 - 20 * log10(M_tip / 0.9)
            end
        else
            if M_tip <= 0.9
                F1IB = 55.5 + 20 * log10(M_d_fan)
            else
                F1IB = 55.5 + 20 * log10(M_d_fan) - 20 * log10(M_tip / 0.9)
            end
        end
        
        # Rotor-stator spacing correction term (F2 of Eqn 4, Figure 6B):
        if settings["fan_id"] == false
            F2IB = -5 * log10(RSS_fan / 300)  # If no distortion
        else
            if RSS_fan <= 100
                F2IB = -5 * log10(RSS_fan / 300)
            else
                F2IB = -5 * log10(100. / 300)  # This is set to a constant 2.3856
            end
        end

    elseif settings["fan_BB_method"] == "geae"
        # Tip Mach-dependent term (F1 of Eqn 4 in report, Figure 4A, modified by GE):
        if M_d_fan <= 1
            if M_tip <= 0.9
                F1IB = 58.5
            else
                F1IB = 58.5 - 50 * log10(M_tip / 0.9)
            end
        else
            if M_tip <= 0.9
                F1IB = 58.5 + 20 * log10(M_d_fan)
            else
                F1IB = 58.5 + 20 * log10(M_d_fan) - 50 * log10(M_tip / 0.9)
            end
        end

        # Rotor-stator spacing correction term (F2 of Eqn 4, Figure 6B):
        F2IB = zeros(T, 1)

    elseif settings["fan_BB_method"] == "kresja"
        # Tip Mach-dependent term (F1, of Eqn 4 in report, Figure 4A, modified by Krejsa):
        if M_d_fan <= 1
            if M_tip < 0.72
                F1IB = 34 + 20 * log10(1. / 1.245)
            else
                F1IB = 34 - 43 * (M_tip - 0.72) + 20 * nlog10(1. / 1.245)
            end
        else
            if M_tip < 0.72
                F1IB = 34 + 20 * log10(M_d_fan / 1.245)
            else
                F1IB = 34 - 43 * (M_tip - 0.72) + 20 * log10(M_d_fan/ 1.245)
            end
        end
        
        # Rotor-stator spacing correction term (F2, of Eqn 4, Figure 6B):
        if settings["fan_id"] == false
            F2IB = -5 * log10(RSS_fan / 300)  # If no distortion
        else
            if RSS_fan <= 100
                F2IB = -5 * log10(RSS_fan / 300)
            else
                F2IB = -5 * log10(100 / 300)  # This is set to a constant 2.3856
            end
        end

    end
    
    # Theta correction term (F3 of Eqn 4, Figure 7A):
    F3IB = pyna_ip.f_F3IB(theta)
    
    # Component value:
    spl_i_b = tsqem .+ F1IB .+ F2IB .+ F3IB

    return spl_i_b
end

function discharge_broadband(pyna_ip, settings, theta, M_tip, tsqem, M_d_fan::Float64, RSS_fan::Float64)

    T = eltype(M_tip)
    # Fan discharge broadband noise component:
    if settings["fan_BB_method"] == "original"
        # Tip Mach-dependent term (F1 of Eqn 10 in report, Figure 4B):
        if M_d_fan <= 1
            if M_tip <= 1
                F1DB = 60
            else
                F1DB = 60 - 20 * log10(M_tip / 1)
            end
        else
            if M_tip <= 1
                F1DB = 60 + 20 * log10(M_d_fan)
            else
                F1DB = 60 + 20 * log10(M_d_fan) - 20 * log10(M_tip / 1)
            end
        end
                
        # Rotor-stator correction term (F2 of Eqn 4, Figure 6B):
        if settings["fan_id"] == false
            F2DB = -5 * log10(RSS_fan / 300)  # If no distortion
        else
            if RSS_fan <= 100
                F2DB = -5 * log10(RSS_fan / 300)
            else
                F2DB = -5 * log10(100 / 300)  # This is set to a constant 2.3856
            end
        end

    elseif settings["fan_BB_method"] == "allied_signal"
        # Tip Mach-dependent term (F1 of Eqn 10 in report, Figure 4B, modified by AlliedSignal):
        if M_d_fan <= 1
            if M_tip <= 1
                F1DB = 58
            else
                F1DB = 58 - 20 * log10(M_tip / 1)
            end
        else
            if M_tip <= 1
                F1DB = 58 + 20 * log10(M_d_fan)
            else
                F1DB = 58 + 20 * log10(M_d_fan) - 20 * log10(M_tip / 1)
            end
        end

        # Rotor-stator spacing correction term (F2 of Eqn 10, Figure 6B, modified by AlliedSignal):
        if settings["fan_id"] == false
            F2DB = -5 * log10(RSS_fan / 300)  # If no distortion
        else
            if RSS_fan <= 100
                F2DB = -5 * log10(RSS_fan / 300)
            else
                F2DB = -5 * log10(100 / 300)  # This is set to a constant 2.3856
            end
        end

    elseif settings["fan_BB_method"] == "geae"
        # Tip Mach-dependent term (F1 of Eqn 10 in report, Figure 4B, modified by GE):
        if M_d_fan <= 1
            if M_tip <= 1
                F1DB = 63
            else
                F1DB = 63 - 30 * log10(M_tip / 1)
            end
        else
            if M_tip <= 1
                F1DB = 63 + 20 * log10(M_d_fan)
            else
                F1DB = 63 + 20 * log10(M_d_fan) - 30 * log10(M_tip / 1)
            end
        end
        
        # Rotor-stator spacing correction term (F2 of Eqn 10, Figure 6B, modified by GE):
        F2DB = -5 * log10(RSS_fan / 300)
    
    elseif settings["fan_BB_method"] == "kresja"
        # Tip Mach-dependent term (F1, of Eqn 10 in report, Figure 4B, modified by Krejsa):
        if M_d_fan <= 1
            # If M_tip < 0.65 Then
            #   F1DBBkrejsa = 34 + 20 * np.log10(1 / 1.245)
            # Else
            F1DB = 34 - 17 * (M_tip - 0.65) + 20 * log10(1 / 1.245)
        else
            # If M_tip < 0.65 Then
            #   F1DBBkrejsa = 34 + 20 * np.log10(M_d_fan / 1.245)
            # Else
            F1DB = 34 - 17 * (M_tip - 0.65) + 20 * log10(M_d_fan / 1.245)
        end
        
        # Rotor-stator spacing correction term (F2, of Eqn 10, Figure 6B):
        if settings["fan_id"] == false
            F2DB = -5 * log10(RSS_fan / 300)  # If no distortion
        else
            if RSS_fan <= 100
                F2DB = -5 * log10(RSS_fan / 300)
            else
                F2DB = -5 * log10(100 / 300)  # This is set to a constant 2.3856
            end
        end
    
    end
        
    # Theta correction term (F3 of Eqn 10, Figure 7B):
    F3DB = pyna_ip.f_F3DB(theta)

    # Added noise factor if there are inlet guide vanes present:
    if settings["fan_igv"] == true
        CDB = 3 * ones(T, 1)
    else
        CDB = zeros(T, 1)
    end

    # Component value:
    spl_d_b = tsqem .+ F1DB .+ F2DB .+ F3DB .+ CDB

    return spl_d_b
end

function inlet_tones(pyna_ip, settings, theta, M_tip, tsqem, M_d_fan::Float64, RSS_fan::Float64)
    
    T = eltype(M_tip)
    # Fan inlet discrete tone noise component:
    if settings["fan_RS_method"] == "original"  # Original method:
        # Tip Mach-dependent term (F1 of Eqn 6 in report, Figure 10A):
        if M_d_fan <= 1
            if M_tip <= 0.72
                F1TI = 60.5
            else
                F1TIA = 60.5 + 50 * log10(M_tip / 0.72)
                F1TIB = 59.5 + 80 * log10(1. / M_tip)
                if F1TIA < F1TIB
                    F1TI = F1TIA
                else
                    F1TI = F1TIB
                end
            end
        else
            if M_tip <= 0.72
                F1TI = 60.5 + 20 * log10(M_d_fan)
            else
                F1TIA = 60.5 + 20 * log10(M_d_fan) + 50 * log10(M_tip / 0.72)
                # Note the 1975 version of NASA TMX-71763 writes it this way:
                # F1TIB = 59.5 + 20 * log10(M_d_fan) + 80 * log10(M_d_fan / M_tip)
                # But the 1979 version of NASA TMX-71763 writes it this way:
                F1TIB = 59.5 + 80 * log10(M_d_fan / M_tip)
                if F1TIA < F1TIB
                    F1TI = F1TIA
                else
                    F1TI = F1TIB
                end
            end
        end

        # Rotor-stator spacing correction term (F2 of Eqn 6, Figure 12):
        if settings["fan_id"] == false
            F2TI = -10 * log10(RSS_fan / 300)  # If no distortion
        else
            if RSS_fan < 100
                F2TI = -10 * log10(RSS_fan / 300)
            else
                F2TI = -10 * log10(100 / 300)  # This is set to a constant 4.7712
            end
        end   
    elseif settings["fan_RS_method"] == "allied_signal"
        # Tip Mach-dependent term (F1 of Eqn 6 in report, Figure 10A, modified by AlliedSignal):
        if M_d_fan <= 1
            if M_tip <= 0.72
                F1TI = 54.5
            else
                F1TIA = 54.5 + 50 * log10(M_tip / 0.72)
                F1TIB = 53.5 + 80 * log10(1. / M_tip)
                if F1TIA < F1TIB
                    F1TI = F1TIA
                else
                    F1TI = F1TIB
                end
            end
        else
            if M_tip <= 0.72
                F1TI = 54.5 + 20 * log10(M_d_fan)
            else
                F1TIA = 54.5 + 20 * log10(M_d_fan) + 50 * log10(M_tip / 0.72)
                F1TIB = 53.5 + 80 * log10(M_d_fan / M_tip)
                if F1TIA < F1TIB
                    F1TI = F1TIA
                else
                    F1TI = F1TIB
                end
            end
        end

        # Rotor-stator spacing correction term (F2 of Eqn 6, Figure 12, modified by AlliedSignal):
        if settings["fan_id"] == false
            F2TI = -10 * log10(RSS_fan / 300)  # If no distortion
        else
            if RSS_fan < 100
                F2TI = -10 * log10(RSS_fan / 300)
            else
                F2TI = -10 * log10(100. / 300)  # This is set to a constant 4.7712
            end
        end
    elseif settings["fan_RS_method"] == "geae"
        # Tip Mach-dependent term (F1 of Eqn 6 in report, Figure 10A, modified by GE):
        if M_d_fan <= 1
            if M_tip <= 0.72
                F1TI = 60.5
            else
                F1TIA = 60.5 + 50 * log10(M_tip / 0.72)
                F1TIB = 64.5 + 80 * log10(1. / M_tip)
                if F1TIA < F1TIB
                    F1TI = F1TIA
                else
                    F1TI = F1TIB
                end
            end
        else
            if M_tip <= 0.72
                F1TI = 60.5 + 20 * log10(M_d_fan)
            else
                F1TIA = 60.5 + 20 * log10(M_d_fan) + 50 * log10(M_tip / 0.72)
                F1TIB = 64.5 + 80 * log10(M_d_fan) - 80 * log10(M_tip)
                if F1TIA < F1TIB
                    F1TI = F1TIA
                else
                    F1TI = F1TIB
                end
            end
        end
        # Rotor-stator spacing correction term (F2 of Eqn 6, Figure 12, modified to zero by GE):
        F2TI = zeros(T, 1)
    elseif settings["fan_RS_method"] == "kresja"
        # Tip Mach-dependent term (F1 of Eqn 6 in report, Figure 10A, modified by Krejsa):
        if M_d_fan <= 1
            F1TI = 42 - 20 * M_tip + 20 * log10(1. / 1.245)
        else
            F1TI = 42 - 20 * M_tip + 20 * log10(M_d_fan / 1.245)
        end

        # Rotor-stator spacing correction term (F2 of Eqn 6, Figure 12):
        if settings["fan_id"] == false
            F2TI = -10 * log10(RSS_fan / 300)  # If no distortion
        else
            if RSS_fan < 100
                F2TI = -10 * log10(RSS_fan / 300)
            else
                F2TI = -10 * log10(100 / 300)  # This is set to a constant 4.7712
            end
        end
    else
        throw(DomainError("Invalid fan_RS_method specified. Specify: original / allied_signal / geae / kresja."))
    end
        
    # Theta correction term (F3 of Eqn 6, Figure 13A):
    F3TI = pyna_ip.f_F3TI(theta)
        
    # Component value:
    tonlv_I = tsqem .+ F1TI .+ F2TI .+ F3TI

    return tonlv_I
end

function discharge_tones(pyna_ip, settings, theta, M_tip, tsqem, M_d_fan::Float64, RSS_fan::Float64)

    T = eltype(M_tip)
    # Fan discharge discrete tone noise component:
    if settings["fan_RS_method"] == "original"  # Original method:
        # Tip Mach-dependent term (F1 of Eqn 12 in report, Figure 10B):
        if M_d_fan <= 1
            if M_tip <= 1
                F1TD = 63
            else
                F1TD = 63 - 20 * log10(M_tip / 1)
            end
        else
            if M_tip <= 1
                F1TD = 63 + 20 * log10(M_d_fan)
            else
                F1TD = 63 + 20 * log10(M_d_fan) - 20 * log10(M_tip / 1)
            end
        end

        # Rotor-stator spacing correction term (F2 of Eqn 12, Figure 12):
        if settings["fan_id"] == false
            F2TD = -10 * log10(RSS_fan / 300)  # If no distortion
        else
            if RSS_fan < 100
                F2TD = -10 * log10(RSS_fan / 300)
            else
                F2TD = -10 * log10(100 / 300)  # This is set to a constant 4.7712
            end
        end
    elseif settings["fan_RS_method"] == "allied_signal"
        # Tip Mach-dependent term (F1 of Eqn 12 in report, Figure 10B, modified by AlliedSignal):
        if M_d_fan <= 1
            if M_tip <= 1
                F1TD = 59
            else
                F1TD = 59 - 20 * log10(M_tip / 1)
            end
        else
            if M_tip <= 1
                F1TD = 59 + 20 * log10(M_d_fan)
            else
                F1TD = 59 + 20 * log10(M_d_fan) - 20 * log10(M_tip / 1)
            end
        end

        # Rotor-stator spacing correction term (F2 of Eqn 12, Figure 12, modified by AlliedSignal):
        if settings["fan_id"] == false
            F2TD = -10 * log10(RSS_fan / 300)  # If no distortion
        else
            if RSS_fan < 100
                F2TD = -10 * log10(RSS_fan / 300)
            else
                F2TD = -10 * log10(100 / 300)  # This is set to a constant 4.7712
            end
        end
    elseif settings["fan_RS_method"] == "geae"
        # Tip Mach-dependent term (F1 of Eqn 12 in report, Figure 10B, modified by GE):
        if M_d_fan <= 1
            if M_tip <= 1
                F1TD = 63
            else
                F1TD = 63 - 20 * log10(M_tip / 1)
            end
        else
            if M_tip <= 1
                F1TD = 63 + 20 * log10(M_d_fan)
            else
                F1TD = 63 + 20 * log10(M_d_fan) - 20 * log10(M_tip / 1)
            end
        end
        # Rotor-stator spacing correction term (F2 of Eqn 12, Figure 12, modified by GE):
        F2TD = -10 * log10(RSS_fan / 300)
    elseif settings["fan_RS_method"] == "kresja"
        # Tip Mach-dependent term (F1 of Eqn 12 in report, Figure 10B, modified by Krejsa):
        if M_d_fan <= 1
            F1TD = 46 - 20 * M_tip + 20 * log10(1 / 1.245)
        else
            F1TD = 46 - 20 * M_tip + 20 * log10(M_d_fan / 1.245)
        end
        # Rotor-stator spacing correction term (F2 of Eqn 12, Figure 12, modified by Krejsa):
        if settings["fan_id"] == false
            F2TD = -10 * log10(RSS_fan / 300)  # If no distortion
        else
            if RSS_fan < 100
                F2TD = -10 * log10(RSS_fan / 300)
            else
                F2TD = -10 * log10(100 / 300)  # This is set to a constant 4.7712
            end
        end
    else
        throw(DomainError("Invalid fan_RS_method specified. Specify: original / allied_signal / geae / kresja."))    
    end
    
    # Theta correction term (F3 of Eqn 6, Figure 13B):
    F3TD = pyna_ip.f_F3TD(theta)

    # Added noise factor if there are inlet guide vanes:
    if settings["fan_igv"] == true
        CDT = 6 * ones(T, 1)
    else
        CDT = zeros(T, 1)
    end

    # Component value:
    tonlv_X = tsqem .+ F1TD .+ F2TD .+ F3TD .+ CDT

    return tonlv_X
end

function combination_tones(pyna_ip, settings, f, theta, M_tip, bpf, tsqem)
    
    # Combination tone (multiple pure tone or buzzsaw) calculations:
    # Note the original Heidmann reference states that MPTs should be computed if
    # the tangential tip speed is supersonic, but the ANOPP implementation states MPTs
    # should be computed if the relative tip speed is supersonic.  The ANOPP implementation
    # is used here, i.e., if M_tip >= 1.0, MPTs are computed.

    # Initialize solution matrices
    T = eltype(M_tip)
    dcp = zeros(T, settings["n_frequency_bands"])

    if M_tip >= 1
        if settings["fan_RS_method"] == "original"
            F2CT = pyna_ip.f_F2CT(theta)

            # Spectrum slopes, original method:
            SL3 = [0, -30, -50, -30]
            SL4 = [0, 30, 50, 50]
            YINT3 = [0, -9.0309, -30.103, -27.0927]
            YINT4 = [0, 9.0309, 30.103, 45.1545]
        elseif settings["fan_RS_method"] == "allied_signal"
            F2CT = pyna_ip.f_F2CT(theta)

            # Spectrum slopes, small fan method:
            SL3 = [0, -15, -50, -30]
            SL4 = [0, 30, 50, 50]
            YINT3 = [0, -4.51545, -30.103, -27.0927]
            YINT4 = [0, 9.0309, 30.103, 45.1545]
        elseif settings["fan_RS_method"] == "geae"
            F2CT = pyna_ip.f_F2CT(theta)

            # Spectrum slopes, GE large fan method:
            SL3 = [0, -30, -50, -30]
            SL4 = [0, 30, 50, 50]
            YINT3 = [0, -9.0309, -30.103, -27.0927]
            YINT4 = [0, 9.0309, 30.103, 45.1545]
        elseif settings["fan_RS_method"] == "kresja"
            F2CT = pyna_ip.f_F2CT(theta)

            # Spectrum slopes, Krejsa method:
            SL3 = [0, -20, -30, -20]
            SL4 = [0, 20, 30, 30]
            YINT3 = [0, -6.0206, -18.0618, -18.0618]
            YINT4 = [0, 6.0206, 18.0618, 27.0927]
        else
            throw(DomainError("Invalid fan_RS_method specified. Specify: original / allied_signal / geae / kresja."))    
        end
        
        # Noise adjustment (reduction) if there are inlet guide vanes, for all methods:
        if settings["fan_igv"] == true
            CCT = -5 * ones(T, 1)
        else
            CCT = zeros(T, 1)
        end
        
        # Loop through the three sub-bpf terms:
        # K = 1; 1/2 bpf term
        # K = 2; 1/4 bpf term
        # K = 3; 1/8 bpf term
        for K in range(1, 3, step=1)
            # Tip Mach-dependent term (F1 of Eqn 8 in Heidmann report, Figure 15A):
            if settings["fan_RS_method"] == "original"
                # Original tip Mach number-dependent term of multiple pure tone noise
                FIG15MT = [0, 1.14, 1.25, 1.61]
                SL1 = [0, 785.68, 391.81, 199.2]
                SL2 = [0, -49.62, -50.06, -49.89]
                YINT1 = [0, 30, 30, 30]
                YINT2 = [0, 79.44, 83.57, 81.52]

                if M_tip < FIG15MT[K]
                    F1CT = SL1[K] * log10(M_tip) + YINT1[K]
                else
                    F1CT = SL2[K] * log10(M_tip) + YINT2[K]
                end
            elseif settings["fan_RS_method"] == "allied_signal"
                # Revised by AlliedSignal:  tip Mach number-dependent term of multiple pure tone noise.
                # Note that a 20log10 independent variable distribution is specified.
                FIG15MT = [0, 1.135, 1.135, 1.413]
                SL1 = [0, 5.9036, 5.54769, 2.43439]
                SL2 = [0, -0.632839, -0.632839, -1.030931]
                YINT1 = [0, 50, 50, 40]
                YINT2 = [0, 57.1896, 56.7981, 50.4058]

                if M_tip < FIG15MT[K]
                    F1CT = SL1[K] * 20 * log10(M_tip) + YINT1[K]
                else
                    F1CT = SL2[K] * 20 * log10(M_tip) + YINT2[K]
                end
            elseif settings["fan_RS_method"] == "geae"
                # Revised by GE: tip Mach number-dependent term of multiple pure tone noise
                FIG15MT = [0, 1.14, 1.25, 1.61]
                SL1 = [0, 746.8608, 398.3077, 118.9406]
                SL2 = [0, -278.96, -284.64, -43.52]
                YINT1 = [0, 30, 30, 36]
                YINT2 = [0, 88.37, 96.18, 69.6]

                if M_tip < FIG15MT[K]
                    F1CT = SL1[K] * log10(M_tip) + YINT1[K]
                else
                    F1CT = SL2[K] * log10(M_tip) + YINT2[K]
                end
            elseif settings["fan_RS_method"] == "kresja"
                FIG15MT = [0, 1.146, 1.322, 1.61]
                SL1 = [0, 785.68, 391.81, 199.2]
                SL2 = [0, -49.62, -50.06, -49.89]
                YINT1 = [0, -18, -15, -12]
                YINT2 = [0, 31.44, 38.57, 39.52]

                if M_tip < FIG15MT[K]
                    F1CT = SL1[K] * log10(M_tip) + YINT1[K]
                else
                    F1CT = SL2[K] * log10(M_tip) + YINT2[K]
                end
            else
                throw(DomainError("Invalid fan_RS_method specified. Specify: original / allied_signal / geae / kresja."))    
            end
                
            CTLC = tsqem + F1CT + F2CT + CCT
            
            # Frequency-dependent term (F3 of Eqn 9, Figure 14):
            FK = 2^K

            # Cycle through frequencies and make assignments:
            for j in range(1, settings["n_frequency_bands"], step=1)
                FQFB = f[j] / bpf

                if FQFB <= 1 / FK
                    # For frequencies less than the subharmonic:
                    F3CT = SL4[K] * log10(FQFB) + YINT4[K]
                else
                    # For frequencies greater than the subharmonic:
                    F3CT = SL3[K] * log10(FQFB) + YINT3[K]
                end
                # Be sure to add the three sub-bpf components together at each frequency:
                dcp[j] = dcp[j] + 10^(0.1 * (CTLC + F3CT))
            end
        end
    end  

    return dcp
end    

function calculate_harmonics(pyna_ip, settings, f, theta, tonlv_I, tonlv_X, i_cut, M_tip, bpf, comp::String)

    # Assign discrete interaction tones at bpf and harmonics to proper bins (see figures 8 and 9):
    # Initialize solution matrices
    T = eltype(M_tip)
    dp = zeros(T, settings["n_frequency_bands"])
    dpx = zeros(T, settings["n_frequency_bands"])

    nfi = 1
    for ih in range(1, settings["n_harmonics"], step=1)

        # Determine the tone fall-off rates per harmonic (harm_i and harm_x):
        if settings["fan_RS_method"] == "original"
            if settings["fan_igv"] == false
                # For fans without inlet guide vanes:
                if i_cut == [0]
                    # For cut-on fans, fundamental:
                    if ih == 1
                        harm_i = zeros(T, 1)
                        harm_x = zeros(T, 1)
                    else
                        # For cut-on fans, harmonics:
                        harm_i = 3 * (ih - 1) * ones(T, 1)
                        harm_x = 3 * (ih - 1) * ones(T, 1)
                    end
                elseif i_cut == [1]
                    # For cut-off fans, fundamental:
                    if ih == 1
                        harm_i = 8 * ones(T, 1)
                        harm_x = 8 * ones(T, 1)
                    else
                        # For cut-off fans, harmonics:
                        harm_i = 3 * (ih - 1) * ones(T, 1)
                        harm_x = 3 * (ih - 1) * ones(T, 1)
                    end
                end
            elseif settings["fan_igv"] == true
                # For fans with inlet guide vanes:
                if i_cut == [0]
                    # For cut-on fans, fundamental:
                    if ih == 1
                        harm_i = zeros(T, 1)
                        harm_x = zeros(T, 1)
                    else
                        # For cut-on fans, harmonics:
                        harm_i = 3 * (ih + 1) * ones(T, 1)
                        harm_x = 3 * (ih + 1) * ones(T, 1)
                    end
                elseif i_cut == [1]
                    # For cut-off fans, fundamental:
                    if ih == 1
                        harm_i = 8 * ones(T, 1)
                        harm_x = 8 * ones(T, 1)
                    else
                        # For cut-off fans, harmonics:
                        harm_i = 3 * (ih + 1) * ones(T, 1)
                        harm_x = 3 * (ih + 1) * ones(T, 1)
                    end
                end
            end
        elseif settings["fan_RS_method"] == "allied_signal"
            if settings["fan_igv"] == false
                # For fans without inlet guide vanes:
                if i_cut == [0]
                    # For cut-on fans, fundamental:
                    if ih == 1
                        harm_i = zeros(T, 1)
                        harm_x = zeros(T, 1)
                    elseif ih == 2
                        # For cut-on fans, second harmonic:
                        harm_i = 9.2 * ones(T, 1)
                        harm_x = 9.2 * ones(T, 1)
                    else
                        # For cut-on fans, upper harmonics:
                        harm_i = (3 * ih + 1.8) * ones(T, 1)
                        harm_x = (3 * ih + 1.8) * ones(T, 1)
                    end
                elseif i_cut == [1]
                    # For cut-off fans, fundamental:
                    if ih == 1
                        harm_i = 8 * ones(T, 1)
                        harm_x = 8 * ones(T, 1)
                    elseif ih == 2
                        # For cut-off fans, second harmonic:
                        harm_i = 9.2 * ones(T, 1)
                        harm_x = 9.2 * ones(T, 1)
                    else
                        # For cut-off fans, upper harmonics:
                        harm_i = (3 * ih + 1.8) * ones(T, 1)
                        harm_x = (3 * ih + 1.8) * ones(T, 1)
                    end
                end
            elseif settings["fan_igv"] == true
                # For fans with inlet guide vanes:
                if i_cut == [0]
                    # For cut-on fans, fundamental:
                    if ih == 1
                        harm_i = zeros(T, 1)
                        harm_x = zeros(T, 1)
                    else
                        # For cut-on fans, harmonics:
                        harm_i = 3 * (ih + 1) * ones(T, 1)
                        harm_x = 3 * (ih + 1) * ones(T, 1)
                    end
                elseif i_cut == [1]
                    # For cut-off fans, fundamental:
                    if ih == 1
                        harm_i = 8 * ones(T, 1)
                        harm_x = 8 * ones(T, 1)
                    else
                        # For cut-off fans, harmonics:
                        harm_i = 3 * (ih + 1) * ones(T, 1)
                        harm_x = 3 * (ih + 1) * ones(T, 1)
                    end
                end
            end
        elseif settings["fan_RS_method"] == "geae"
            if settings["fan_igv"] == false
                # For fans without inlet guide vanes:
                if i_cut == [0]
                    # For cut-on fans, fundamental:
                    if ih == 1
                        harm_i = zeros(T, 1)
                        harm_x = zeros(T, 1)
                    else
                        # For cut-on fans, harmonics:
                        if M_tip < 1.15
                            harm_i = 6 * (ih - 1) * ones(T, 1)
                        else
                            harm_i = 9 * (ih - 1) * ones(T, 1)
                        end
                        harm_x = 3 * (ih - 1) * ones(T, 1)
                    end
                elseif i_cut == [1]
                    # For cut-off fans, fundamental:
                    if ih == 1
                        harm_i = 8 * ones(T, 1)
                        harm_x = 8 * ones(T, 1)
                    else
                        # For cut-off fans, harmonics:
                        if M_tip < 1.15
                            harm_i = 6 * (ih - 1) * ones(T, 1)
                        else
                            harm_i = 9 * (ih - 1) * ones(T, 1)
                        harm_x = 3 * (ih - 1) * ones(T, 1)
                        end
                    end
                end
            elseif settings["fan_igv"] == true
                # For fans with inlet guide vanes:
                if i_cut == [0]
                    # For cut-on fans, fundamental:
                    if ih == 1
                        harm_i = zeros(T, 1)
                        harm_x = zeros(T, 1)
                    else
                        # For cut-on fans, harmonics:
                        harm_i = 3 * (ih + 1) * ones(T, 1)
                        harm_x = 3 * (ih + 1) * ones(T, 1)
                    end
                elseif i_cut == [1]
                    # For cut-off fans, fundamental:
                    if ih == 1
                        harm_i = 8 * ones(T, 1)
                        harm_x = 8 * ones(T, 1)
                    else
                        # For cut-off fans, harmonics:
                        harm_i = 3 * (ih + 1) * ones(T, 1)
                        harm_x = 3 * (ih + 1) * ones(T, 1)
                    end
                end
            end
        elseif settings["fan_RS_method"] == "kresha"
            if settings["fan_igv"] == false
                # For fans without inlet guide vanes:
                if i_cut == [0]
                    # For cut-on fans, fundamental:
                    if ih == 1
                        harm_i = zeros(T, 1)
                        harm_x = zeros(T, 1)
                    else
                        # For cut-on fans, harmonics:
                        harm_i = 3 * (ih - 1) * ones(T, 1)
                        harm_x = 3 * (ih - 1) * ones(T, 1)
                    end
                elseif i_cut == [1]
                    # For cut-off fans, fundamental:
                    if ih == 1
                        harm_i = 8 * ones(T, 1)
                        harm_x = 8 * ones(T, 1)
                    else
                        # For cut-off fans, harmonics:
                        harm_i = 3 * (ih - 1) * ones(T, 1)
                        harm_x = 3 * (ih - 1) * ones(T, 1)
                    end
                end
            elseif settings["fan_igv"] == true
                # For fans with inlet guide vanes:
                if i_cut == [0]
                    # For cut-on fans, fundamental:
                    if ih == 1
                        harm_i = zeros(T, 1)
                        harm_x = zeros(T, 1)
                    else
                        # For cut-on fans, harmonics:
                        harm_i = 3 * (ih + 1) * ones(T, 1)
                        harm_x = 3 * (ih + 1) * ones(T, 1)
                    end
                elseif i_cut == [1]
                    # For cut-off fans, fundamental:
                    if ih == 1
                        harm_i = 8 * ones(T, 1)
                        harm_x = 8 * ones(T, 1)
                    else
                        # For cut-off fans, harmonics:
                        harm_i = 3 * (ih + 1) * ones(T, 1)
                        harm_x = 3 * (ih + 1) * ones(T, 1)
                    end
                end
            end
        else
            throw(DomainError("Invalid fan_RS_method specified. Specify: original / allied_signal / geae / kresja."))    
        end
        # Calculate TCS and distor
        if (settings["fan_id"] == true) && (settings["fan_RS_method"] != "geae")
            # Assign the increment to the fundamental tone along with a 10 dB per harmonic order fall-off
            # for cases with inlet flow distortion (see figure 9):
            distor = 10^(0.1 * tonlv_I - ih + 1)
            TCS = zeros(T, 1)
        elseif (settings["fan_id"] == true) && (settings["fan_RS_method"] == "geae")
            # Compute suppression factors for GE#s "Flight cleanup Turbulent Control Structure."
            # Approach or takeoff values to be applied to inlet discrete interaction tones
            # at bpf and 2bpf.  Accounts for observed in-flight tendencies.

            if settings["fan_ge_flight_cleanup"] == "takeoff"
                # Apply takeoff values:
                if ih == 1
                    TCS = pyna_ip.f_TCS_takeoff_ih1(theta)
                elseif ih == 2
                    TCS = pyna_ip.f_TCS_takeoff_ih2(theta)
                else
                    TCS = zeros(T, 1)
                end
            elseif settings["fan_ge_flight_cleanup"] == "approach"
                # Apply approach values:
                if ih == 1
                    TCS = pyna_ip.f_TCS_approach_ih1(theta)
                elseif ih == 2
                    TCS = pyna_ip.f_TCS_approach_ih2(theta)
                else
                    TCS = zeros(T, 1)
                end
            elseif settings["fan_ge_flight_cleanup"] == "none"
                # Apply zero values (i.e., fan inlet flow is distorted):
                TCS = zeros(T, 1)
            end
            # Inlet distortion effects are always included in basic inlet tone model of the GE method.
            # Flight cleanup levels are then subtracted from the inlet tones if the flow is not distorted.
            # The flight cleanup levels are set to zero if the flow is distorted.
            # Use the same increment as the original method and the same 10 dB per harmonic fall-off rate.
            distor = 10 .^(0.1 * (tonlv_I .- TCS) - ih + 1)
        else
            distor = zeros(T, 1)
            TCS = zeros(T, 1)
        end
        
        # Calculate tone power
        if comp == "fan_inlet" # or comp == "inlet RS":
            tonpwr_i = 10 .^(0.1 * (tonlv_I .- harm_i .- TCS)) .+ distor
        else
            tonpwr_i = zeros(T, 1)
        end
        if comp == "fan_discharge" # or comp == "discharge RS" or comp == "total":
            tonpwr_x = 10 .^(0.1 * (tonlv_X .- harm_x))
        else
            tonpwr_x = zeros(T, 1)
        end
                
        # Compute filter bandwidths:
        filbw = 1  # Fraction of filter bandwidth with gain of unity (default to unity)
        F1 = 0.78250188 + 0.10874906 * filbw
        F2 = 1 - 0.10874906 * filbw
        F3 = 1 + 0.12201845 * filbw
        F4 = 1.2440369 - 0.12201845 * filbw

        # Cycle through frequencies and assign tones to 1/3rd octave bins:
        ll = NaN
        for l in range(nfi, settings["n_frequency_bands"], step=1)
            Frat = bpf * ih / f[l]
            FR = 1
            if Frat .< F1
                break
            elseif Frat .> F4
                ll = l
                continue
            #elseif Frat .> F3
                #FR = (F4 .- Frat) ./ (F4 .- F3)
            #elseif Frat .< F2
                #FR = (Frat .- F1) ./ (F2 .- F1)
            end

            dp[l] = dp[l] .+ tonpwr_i[1] .* FR
            dpx[l] = dpx[l] .+ tonpwr_x[1] .* FR
            nfi = ll

            #continue
        end
    end

    return dp, dpx
end

function calculate_cutoff(M_tip_tan, B_fan::Int64, V_fan::Int64)
    
    # Vane/blade ratio parameter:
    vane_blade_ratio = 1 - V_fan / B_fan
    if vane_blade_ratio == 0
        vane_blade_ratio = 1e-6
    end
    
    # Cutoff parameter:
    cutoff = abs(M_tip_tan / vane_blade_ratio)

    # if the cutoff parameter is less than 1.05 and the tip Mach is less than
    # unity, the fan is cut off (i.e., the tones are reduced in magnitude):
    if (cutoff < 1.05) && (M_tip_tan < 1)
        i_cut = ones(eltype(cutoff), (1, ))
    elseif (cutoff < 1.05) && (M_tip_tan >= 1)
        i_cut = zeros(eltype(cutoff), (1, ))
    elseif (cutoff >= 1.05) && (M_tip_tan < 1)
        i_cut = zeros(eltype(cutoff), (1, ))
    else
        i_cut = zeros(eltype(cutoff), (1, ))
    end

    return i_cut
end

function fan_source!(spl, pyna_ip, settings, ac, f, shield, M_0, c_0, T_0, rho_0, theta, DTt_f_star, mdot_f_star, N_f_star, A_f_star, d_f_star, comp::String)

    # Initialize solution
    T = eltype(DTt_f_star)
    
    ### Extract the inputs
    delta_T_fan = DTt_f_star * T_0  # Total temperature rise across fan [R]
    rpm = N_f_star * 60 * c_0 / (d_f_star * sqrt(settings["A_e"]))  # Shaft speed [rpm]
    M_tip_tan = (d_f_star * sqrt(settings["A_e"]) / 2) * rpm * 2 * π / 60 / c_0  # Tangential (i.e., radius*omega) tip Mach number: ! Doesn"t need correction
    mdot_fan = mdot_f_star * rho_0 * c_0 * settings["A_e"]  # Airflow [kg/s]
    bpf = rpm * ac.B_fan / 60. / (1 .- M_0 * cos.(theta * π / 180))  # Blade passing frequency, [Hz]
    flow_M = mdot_fan / (rho_0 .* A_f_star * settings["A_e"] * c_0)  # Fan face flow Mach number (assumes ambient and fan face static densities are equal): !!!!!!!
    M_tip = (M_tip_tan^2 + flow_M^2)^0.5  # Relative (i.e., helical) tip Mach number: ! Doesn"t need correction

    # Temperature-flow power base term:
    rho_sl = 1.22514
    c_sl = 340.29395
    if settings["fan_BB_method"] == "kresja"
        tsqem = 40 * log10(delta_T_fan * 1.8) + 10 * log10(2.20462 * mdot_fan / (1 - M_0 * cos(theta * π / 180)).^4) - 20 * log10(settings["r_0"]) - 10*log10(rho_sl^2 * c_sl^4)
    else  # All other methods:
        tsqem = 20 * log10(delta_T_fan * 1.8) + 10 * log10(2.20462 * mdot_fan / (1 - M_0 * cos(theta * π / 180)).^4) - 20 * log10(settings["r_0"]) - 10*log10(rho_sl^2 * c_sl^4)
    end

    # Calculate individual noise components
    if comp == "fan_inlet"
        spl_i_b = inlet_broadband(pyna_ip, settings, theta, M_tip, tsqem, ac.M_d_fan, ac.RSS_fan)
        spl_d_b = 0.
        tonlv_I = inlet_tones(pyna_ip, settings, theta, M_tip, tsqem, ac.M_d_fan, ac.RSS_fan)
        tonlv_X = 0.
    elseif comp == "fan_discharge"
        spl_i_b = 0.
        spl_d_b = discharge_broadband(pyna_ip, settings, theta, M_tip, tsqem, ac.M_d_fan, ac.RSS_fan)
        tonlv_I = 0.
        tonlv_X = discharge_tones(pyna_ip, settings, theta, M_tip, tsqem, ac.M_d_fan, ac.RSS_fan)
    end
            
    if settings["fan_combination_tones"]
        dcp = combination_tones(pyna_ip, settings,  f, theta, M_tip, bpf, tsqem)
    else
        dcp = zeros(T, settings["n_frequency_bands"])
    end

    # Calculate if cut-off happens (1) or not (0)
    i_cut = calculate_cutoff(M_tip_tan, ac.B_fan, ac.V_fan)

    # Assign tones_to bands
    dp, dpx = calculate_harmonics(pyna_ip, settings, f, theta, tonlv_I, tonlv_X, i_cut, M_tip, bpf, comp)

    # Final calculations;  cycle through frequencies and assign values:
    if settings["fan_BB_method"] == "allied_signal"
        # Eqn 2 or Figure 3A:
        # if f[j] / bpf < 2:
        # flog_i,exit = -10 * np.log10(np.exp(-0.35 * (np.log(f[j] / bpf / 2.0) / np.log(2.2))^2))
        flog_i = 2.445096095 * (log.(f / bpf / 2)).^2
        # elif f[j] / bpf > 2:
        # flog_i,exit = -10 * np.log10(np.exp(-2.0 * (np.log(f[j] / bpf / 2.0) / np.log(2.2))^2))
        flog_i[f/bpf .> 2] = (13.97197769 * (log.(f / bpf / 2)).^2)[f/bpf .> 2]
        flog_e = flog_i

    elseif settings["fan_BB_method"] == "kresja"
        # Eqn 2 or Figure 3A:
        # flog_i = -10 * np.log10(np.exp(-0.5 * (np.log(f[j] / bpf / 4) / np.log(2.2))^2))
        # Which may be simplified as:
        flog_i = 3.4929944 * (log.(f / bpf / 4)).^2
        # flog_e = -10 * np.log10(np.exp(-0.5 * (np.log(f[j] / bpf / 2.5) / np.log(2.2))^2))
        # Which may be simplified as:
        flog_e = 3.4929944 * (log.(f / bpf / 2.5)).^2
    else
        # For the original or the GE large fan methods:
        # Eqn 2 or Figure 3A:
        # flog_i,exit = -10 * np.log10(np.exp(-0.5 * (np.log(f[j] / bpf / 2.5) / np.log(2.2))^2))
        # Which may be simplified as:
        flog_i = 3.4929944 * (log.(f ./ bpf / 2.5)).^2
        flog_e = flog_i
    end

    if comp == "fan_inlet"
        pow_level_fan = 10 .^(0.1 * (spl_i_b .- flog_i))
        pow_level_fan = pow_level_fan .+ dp

    # Add discrete tone and broadband components for exhaust noise:
    elseif comp == "fan_discharge"
        pow_level_fan = 10 .^(0.1 * (spl_d_b .- flog_e))
        pow_level_fan = pow_level_fan .+ dpx
    else
        throw(DomainError("Invalid component specified."))
    end
        
    # Add inlet combination tones if needed:
    if (M_tip > 1) && (settings["fan_combination_tones"] == true)
        pow_level_fan = pow_level_fan .+ dcp
    end
        
    # Add ambient correction. Multiply with number of engines
    msap_j = ac.n_eng * pow_level_fan * settings["p_ref"]^2

    # Fan liner suppression
    if settings["fan_liner_suppression"]
        if comp == "fan_inlet"
            supp = pyna_ip.f_supp_fi.(f, theta.*ones(T, settings["n_frequency_bands"], ))
        elseif comp == "fan_discharge"
            supp = pyna_ip.f_supp_fd.(f, theta.*ones(T, settings["n_frequency_bands"], ))
        end
        msap_j = supp .* msap_j
    end

    # Fan inlet shielding 
    if (settings["shielding"]==true) && (comp == "fan_inlet")
        msap_j = msap_j ./ (10 .^(shield / 10))
    end

    # Normalize msap by reference pressure
    @. spl += clamp.(msap_j, 1e-99, Inf)/settings["p_ref"]^2

end