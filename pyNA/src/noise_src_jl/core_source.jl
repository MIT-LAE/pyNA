using ReverseDiff

function core_source_ge!(spl::Array, x::Union{Array, ReverseDiff.TrackedArray}, settings, pyna_ip, af, f::Array{Float64,1})

    # x = [mdoti_c, Tti_c, Ttj_c, Pti_c, DTt_des_c, c_0, rho_0, T_0, p_0, M_0, TS, theta]
    # y = spl
    
    # normalize engine variables
    mdoti_c_star = x[1] / (x[7] * x[6] * settings["A_e"])
    Tti_c_star = x[2] / x[8]
    Ttj_c_star = x[3] / x[8]
    Pti_c_star = x[4] / x[9]
    DTt_des_c_star = x[5] / x[8]

    # Initialize solution
    r_s_star = settings["r_0"]/sqrt(settings["A_e"])
    A_c_star = 1.

    # Turbine transmission loss function
    # Source: Zorumski report 1982 part 2. Chapter 8.2 Equation 3
    g_TT = DTt_des_c_star^(-4)
    
    # Calculate acoustic power (Pi_star)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 3
    Pi_star = 8.85e-7 * (mdoti_c_star / A_c_star) * ((Ttj_c_star - Tti_c_star) / Tti_c_star)^2 * Pti_c_star^2 * g_TT
    
    # Calculate directivity function (D)
    D_funct = 10 ^pyna_ip.f_D_core(x[12])

    # Calculate the spectral function (S)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 4
    f_p = 400 / (1 - x[10] * cos.(x[12] * π / 180.))
    log10ffp = log10.(f / f_p)
    S_funct = 10 .^pyna_ip.f_S_core.(log10ffp)
      
    # # Calculate mean-square acoustic pressure (msap)
    # # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 1
    # # Multiply with number of engines and normalize msap by reference pressure
    pow_level_core = af.n_eng / settings["p_ref"]^2 * Pi_star * A_c_star / (4 * π * r_s_star^2) * D_funct / (1 - x[10] * cos(π / 180. * x[12]))^4
    if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"] && x[11] > 0.8
        spl .+= pow_level_core * 10 ^(-2.3 / 10.) * S_funct
    else
        spl .+= pow_level_core * S_funct
    end
end


function core_source_pw!(spl::Array, x::Union{Array, ReverseDiff.TrackedArray}, settings, pyna_ip, af, f::Array{Float64, 1})
    
    # x = [mdoti_c, Tti_c, Ttj_c, Pti_c, rho_te_c, c_te_c, rho_ti_c, c_ti_c, c_0, T_0, rho_0, p_0, M_0, TS, theta]
    # y = spl

    # Normalize engine variables
    mdoti_c_star = x[1] / (x[9] * settings["A_e"] * x[11])
    Tti_c_star = x[2] / x[10]
    Ttj_c_star = x[3] / x[10]
    Pti_c_star = x[4] / x[12]
    rho_te_c_star = x[5] / x[11]
    c_te_c_star = x[6] / x[9]
    rho_ti_c_star = x[7] / x[11]
    c_ti_c_star = x[8] / x[9]

    # Initialize solution
    r_s_star = settings["r_0"]/sqrt(settings["A_e"])
    A_c_star = 1.

    # Turbine transmission loss function
    # Source: Hultgren, 2012: A comparison of combustor models Equation 6
    zeta = (rho_te_c_star*c_te_c_star) / (rho_ti_c_star*c_ti_c_star)
    g_TT = 0.8 * zeta / (1 + zeta)^2
    
    # Calculate acoustic power (Pi_star)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 3
    Pi_star = 8.85e-7 * (mdoti_c_star / A_c_star) * ((Ttj_c_star - Tti_c_star) / Tti_c_star)^2 * Ptj_c_star^2 * g_TT
    
    # Calculate directivity function (D)
    D_funct = 10 ^pyna_ip.f_D_core(x[15])

    # Calculate the spectral function (S)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 4
    f_p = 400 / (1 - x[13] * cos(x[15] * π / 180.))
    log10ffp = log10.(f / f_p)

    # Take the S function as SAE ARP876E Table 17 and all other values which are not in the table from Zorumski report 1982 part 2. Chapter 8.2 Table III
    S_funct = 10 .^pyna_ip.f_S_core.(log10ffp)

    # Calculate mean-square acoustic pressure (msap)
    # Source Zorumski report 1982 part 2. Chapter 8.2 Equation 1
    # Multiply with number of engines and normalize msap by reference pressure
    pow_level_core = af.n_eng / settings["p_ref"]^2 * Pi_star * A_c_star / (4 * π * r_s_star^2) * D_funct / (1. - x[13] * cos.(π / 180. * x[15]))^4
    if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"] && x[14] > 0.8
        spl .+= pow_level_core * 10 ^(-2.3 / 10.) * S_funct
    else
        spl .+= pow_level_core * S_funct
    end
end

core_source_ge_fwd! = (y,x)->core_source_ge!(y, x, settings, pyna_ip, ac, f)
core_source_pw_fwd! = (y,x)->core_source_pw!(y, x, settings, pyna_ip, ac, f)