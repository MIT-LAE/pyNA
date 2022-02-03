using OpenMDAO: AbstractExplicitComp, VarData, PartialsData, get_rows_cols
import OpenMDAO: setup, compute!, compute_partials!
using Interpolations
using ReverseDiff
using LinearAlgebra
using ConcreteStructs
using PCHIPInterpolation
using Dates

include("normalization_engine_variables.jl")
include("shielding.jl")
include("geometry.jl")
include("source.jl")
include("fan.jl")
include("core.jl")
include("jet.jl")
include("airframe.jl")
include("propagation.jl")
include("split_subbands.jl")
include("lateral_attenuation.jl")
include("ground_reflections.jl")
include("spl.jl")
include("oaspl.jl")
include("pnlt.jl")
include("epnl.jl")
include("ipnlt.jl")
include("ioaspl.jl")

# Define propagation struct
@concrete struct Noise <: AbstractExplicitComp
    settings
    data
    ac
    n_t
    idx
    idx_src
    optimization
    noise
    X
    J
    noise_fwd
    noise_tape
end

function Noise(settings, data, ac, n_t::Int, idx, idx_src, optimization)

    function noise(settings, data, ac, n_t, idx, idx_src, optimization, input_v)

        # Unpack inputs
        x = input_v[idx["x"][1]:idx["x"][2]]
        y = input_v[idx["y"][1]:idx["y"][2]]
        z = input_v[idx["z"][1]:idx["z"][2]]
        alpha = input_v[idx["alpha"][1]:idx["alpha"][2]]
        gamma = input_v[idx["gamma"][1]:idx["gamma"][2]]
        t_s = input_v[idx["t_s"][1]:idx["t_s"][2]]
        rho_0 = input_v[idx["rho_0"][1]:idx["rho_0"][2]]
        mu_0 = input_v[idx["mu_0"][1]:idx["mu_0"][2]]
        c_0 = input_v[idx["c_0"][1]:idx["c_0"][2]]
        T_0 = input_v[idx["T_0"][1]:idx["T_0"][2]]
        p_0 = input_v[idx["p_0"][1]:idx["p_0"][2]]
        M_0 = input_v[idx["M_0"][1]:idx["M_0"][2]]
        I_0 = input_v[idx["I_0"][1]:idx["I_0"][2]]
        TS = input_v[idx["TS"][1]:idx["TS"][2]]
        
        if settings.jet_mixing == true && settings.jet_shock == false
            V_j = input_v[idx["V_j"][1]:idx["V_j"][2]]
            rho_j = input_v[idx["rho_j"][1]:idx["rho_j"][2]]
            A_j = input_v[idx["A_j"][1]:idx["A_j"][2]]
            Tt_j = input_v[idx["Tt_j"][1]:idx["Tt_j"][2]]
        elseif settings.jet_shock == true && settings.jet_mixing == false
            V_j = input_v[idx["V_j"][1]:idx["V_j"][2]]
            M_j = input_v[idx["M_j"][1]:idx["M_j"][2]]
            A_j = input_v[idx["A_j"][1]:idx["A_j"][2]]
            Tt_j = input_v[idx["Tt_j"][1]:idx["Tt_j"][2]]
        elseif settings.jet_shock ==true && settings.jet_mixing == true
            V_j = input_v[idx["V_j"][1]:idx["V_j"][2]]
            rho_j = input_v[idx["rho_j"][1]:idx["rho_j"][2]]
            A_j = input_v[idx["A_j"][1]:idx["A_j"][2]]
            Tt_j = input_v[idx["Tt_j"][1]:idx["Tt_j"][2]]
            M_j = input_v[idx["M_j"][1]:idx["M_j"][2]]
        end
        if settings.core
            if settings.method_core_turb == "GE"
                mdoti_c = input_v[idx["mdoti_c"][1]:idx["mdoti_c"][2]]
                Tti_c = input_v[idx["Tti_c"][1]:idx["Tti_c"][2]]
                Ttj_c = input_v[idx["Ttj_c"][1]:idx["Ttj_c"][2]]
                Pti_c = input_v[idx["Pti_c"][1]:idx["Pti_c"][2]]
                DTt_des_c = input_v[idx["DTt_des_c"][1]:idx["DTt_des_c"][2]]
            elseif settings.method_core_turb == "PW"
                mdoti_c = input_v[idx["mdoti_c"][1]:idx["mdoti_c"][2]]
                Tti_c = input_v[idx["Tti_c"][1]:idx["Tti_c"][2]]
                Ttj_c = input_v[idx["Ttj_c"][1]:idx["Ttj_c"][2]]
                Pti_c = input_v[idx["Pti_c"][1]:idx["Pti_c"][2]]
                rho_te_c = input_v[idx["rho_te_c"][1]:idx["rho_te_c"][2]]
                c_te_c = input_v[idx["c_te_c"][1]:idx["c_te_c"][2]]
                rho_ti_c = input_v[idx["rho_ti_c"][1]:idx["rho_ti_c"][2]]
                c_ti_c = input_v[idx["c_ti_c"][1]:idx["c_ti_c"][2]]
            end
        end
        if settings.airframe
            theta_flaps = input_v[idx["theta_flaps"][1]:idx["theta_flaps"][2]]
            I_landing_gear = input_v[idx["I_landing_gear"][1]:idx["I_landing_gear"][2]]
        end
        if settings.fan_inlet==true || settings.fan_discharge==true
            DTt_f = input_v[idx["DTt_f"][1]:idx["DTt_f"][2]]
            mdot_f = input_v[idx["mdot_f"][1]:idx["mdot_f"][2]]
            N_f = input_v[idx["N_f"][1]:idx["N_f"][2]]
            A_f = input_v[idx["A_f"][1]:idx["A_f"][2]]
            d_f = input_v[idx["d_f"][1]:idx["d_f"][2]]
        end

        # Get type of input vector
        T = eltype(input_v)

        # Number of observers
        n_obs = size(settings.x_observer_array)[1]

        # Compute noise for each observer
        t_o = zeros(eltype(input_v), (n_obs, n_t))
        msap_source = zeros(eltype(input_v), (n_obs, n_t, settings.N_f))
        msap_prop = zeros(eltype(input_v), (n_obs, n_t, settings.N_f))
        Noy = zeros(eltype(input_v), (n_obs, n_t, settings.N_f))
        spl = zeros(eltype(input_v), (n_obs, n_t, settings.N_f))
        oaspl = zeros(eltype(input_v), (n_obs, n_t))
        pnlt = zeros(eltype(input_v), (n_obs, n_t))
        C = zeros(eltype(input_v), (n_obs, n_t, settings.N_f))
        level_int = zeros(eltype(input_v), (n_obs, ))

        # Compute airframe shielding coefficients
        if settings.shielding == true
            shield = shielding(settings, n_t)
        end

        # Iterate over observers
        for i in range(1, n_obs, step=1)

            println("Computing noise at observer: ", i)

            # Extract observer location
            x_obs = settings.x_observer_array[i,:]

            # Compute geometry 
            input_geom = x
            input_geom = vcat(input_geom, y)
            input_geom = vcat(input_geom, z)
            input_geom = vcat(input_geom, alpha)
            input_geom = vcat(input_geom, gamma)
            input_geom = vcat(input_geom, t_s)
            input_geom = vcat(input_geom, c_0)
            input_geom = vcat(input_geom, T_0)
            output_geom = geometry(settings, x_obs, n_t, input_geom)
            r = output_geom[0*n_t + 1 : 1*n_t]
            beta = output_geom[1*n_t + 1 : 2*n_t]
            theta =  output_geom[2*n_t + 1 : 3*n_t]
            phi = output_geom[3*n_t + 1 : 4*n_t]
            c_bar = output_geom[4*n_t + 1 : 5*n_t]
            t_o[i,:] = output_geom[5*n_t + 1 : 6*n_t]

            # Normalize engine inputs
            if settings.jet_mixing == true && settings.jet_shock == false
                input_norm = V_j
                input_norm = vcat(input_norm, rho_j)
                input_norm = vcat(input_norm, A_j)
                input_norm = vcat(input_norm, Tt_j)
                input_norm = vcat(input_norm, c_0)
                input_norm = vcat(input_norm, rho_0)
                input_norm = vcat(input_norm, T_0)
                output_norm = normalization_engine_variables(settings, n_t, input_norm, "jet_mixing")
                V_j_star = output_norm[0*n_t + 1 : 1*n_t]
                rho_j_star = output_norm[1*n_t + 1 : 2*n_t]
                A_j_star = output_norm[2*n_t + 1 : 3*n_t]
                Tt_j_star = output_norm[3*n_t + 1 : 4*n_t]
            elseif settings.jet_shock == true && settings.jet_mixing == false
                input_norm = V_j
                input_norm = vcat(input_norm, A_j)
                input_norm = vcat(input_norm, Tt_j)
                input_norm = vcat(input_norm, c_0)
                input_norm = vcat(input_norm, T_0)
                output_norm = normalization_engine_variables(settings, n_t, input_norm, "jet_shock")
                V_j_star = output_norm[0*n_t + 1 : 1*n_t]
                A_j_star = output_norm[1*n_t + 1 : 2*n_t]
                Tt_j_star = output_norm[2*n_t + 1 : 3*n_t]
            elseif settings.jet_shock ==true && settings.jet_mixing == true
                input_norm = V_j
                input_norm = vcat(input_norm, rho_j)
                input_norm = vcat(input_norm, A_j)
                input_norm = vcat(input_norm, Tt_j)
                input_norm = vcat(input_norm, c_0)
                input_norm = vcat(input_norm, rho_0)
                input_norm = vcat(input_norm, T_0)
                output_norm = normalization_engine_variables(settings, n_t, input_norm, "jet")
                V_j_star = output_norm[0*n_t + 1 : 1*n_t]
                rho_j_star = output_norm[1*n_t + 1 : 2*n_t]
                A_j_star = output_norm[2*n_t + 1 : 3*n_t]
                Tt_j_star = output_norm[3*n_t + 1 : 4*n_t]
            end
            if settings.core
                if settings.method_core_turb == "GE"
                    input_norm = mdoti_c
                    input_norm = vcat(input_norm, Tti_c)
                    input_norm = vcat(input_norm, Ttj_c)
                    input_norm = vcat(input_norm, Pti_c)
                    input_norm = vcat(input_norm, DTt_des_c)
                    input_norm = vcat(input_norm, c_0)
                    input_norm = vcat(input_norm, rho_0)
                    input_norm = vcat(input_norm, T_0)
                    input_norm = vcat(input_norm, p_0)
                    output_norm = normalization_engine_variables(settings, n_t, input_norm, "core_ge")
                    mdoti_c_star = output_norm[0*n_t + 1 : 1*n_t]
                    Tti_c_star = output_norm[1*n_t + 1 : 2*n_t]
                    Ttj_c_star = output_norm[2*n_t + 1 : 3*n_t]
                    Pti_c_star = output_norm[3*n_t + 1 : 4*n_t]
                    DTt_des_c_star = output_norm[4*n_t + 1 : 5*n_t]
                elseif settings.method_core_turb == "PW"
                    input_norm = mdoti_c
                    input_norm = vcat(input_norm, Tti_c)
                    input_norm = vcat(input_norm, Ttj_c)
                    input_norm = vcat(input_norm, Pti_c)
                    input_norm = vcat(input_norm, rho_te_c)
                    input_norm = vcat(input_norm, c_te_c)
                    input_norm = vcat(input_norm, rho_ti_c)
                    input_norm = vcat(input_norm, c_ti_c)
                    input_norm = vcat(input_norm, c_0)
                    input_norm = vcat(input_norm, rho_0)
                    input_norm = vcat(input_norm, T_0)
                    input_norm = vcat(input_norm, p_0)
                    output_norm = normalization_engine_variables(settings, n_t, input_norm, "core_pw")
                    mdoti_c_star = output_norm[0*n_t + 1 : 1*n_t]
                    Tti_c_star = output_norm[1*n_t + 1 : 2*n_t]
                    Ttj_c_star = output_norm[2*n_t + 1 : 3*n_t]
                    Pti_c_star = output_norm[3*n_t + 1 : 4*n_t]
                    rho_te_c_star = output_norm[4*n_t + 1 : 5*n_t]
                    c_te_c_star = output_norm[5*n_t + 1 : 6*n_t]
                    rho_ti_c_star = output_norm[6*n_t + 1 : 7*n_t]
                    c_ti_c_star = output_norm[7*n_t + 1 : 8*n_t]
                end
            end
            if settings.fan_inlet==true || settings.fan_discharge==true
                input_norm = DTt_f
                input_norm = vcat(input_norm, mdot_f)
                input_norm = vcat(input_norm, N_f)
                input_norm = vcat(input_norm, A_f)
                input_norm = vcat(input_norm, d_f)
                input_norm = vcat(input_norm, c_0)
                input_norm = vcat(input_norm, rho_0)
                input_norm = vcat(input_norm, T_0)
                output_norm = normalization_engine_variables(settings, n_t, input_norm, "fan")
                DTt_f_star = output_norm[0*n_t + 1 : 1*n_t]
                mdot_f_star = output_norm[1*n_t + 1 : 2*n_t]
                N_f_star = output_norm[2*n_t + 1 : 3*n_t]
                A_f_star = output_norm[3*n_t + 1 : 4*n_t]
                d_f_star = output_norm[4*n_t + 1 : 5*n_t]
            end

            # Compute source
            input_src = TS
            input_src = vcat(input_src, M_0)
            input_src = vcat(input_src, c_0)
            input_src = vcat(input_src, rho_0)
            input_src = vcat(input_src, mu_0)
            input_src = vcat(input_src, T_0)
            input_src = vcat(input_src, theta)
            input_src = vcat(input_src, phi)
            if settings.jet_mixing == true && settings.jet_shock == false
                input_src = vcat(input_src, V_j_star)
                input_src = vcat(input_src, rho_j_star)
                input_src = vcat(input_src, A_j_star)
                input_src = vcat(input_src, Tt_j_star)
            elseif settings.jet_shock == true && settings.jet_mixing == false
                input_src = vcat(input_src, V_j_star)
                input_src = vcat(input_src, M_j)
                input_src = vcat(input_src, A_j_star)
                input_src = vcat(input_src, Tt_j_star)
            elseif settings.jet_shock ==true && settings.jet_mixing == true
                input_src = vcat(input_src, V_j_star)
                input_src = vcat(input_src, rho_j_star)
                input_src = vcat(input_src, A_j_star)
                input_src = vcat(input_src, Tt_j_star)
                input_src = vcat(input_src, M_j)
            end
            if settings.core
                if settings.method_core_turb == "GE"
                    input_src = vcat(input_src, mdoti_c_star)
                    input_src = vcat(input_src, Tti_c_star)
                    input_src = vcat(input_src, Ttj_c_star)
                    input_src = vcat(input_src, Pti_c_star)
                    input_src = vcat(input_src, DTt_des_c_star)
                elseif settings.method_core_turb == "PW"
                    input_src = vcat(input_src, mdoti_c_star)
                    input_src = vcat(input_src, Tti_c_star)
                    input_src = vcat(input_src, Ttj_c_star)
                    input_src = vcat(input_src, Pti_c_star)
                    input_src = vcat(input_src, rho_te_c_star)
                    input_src = vcat(input_src, c_te_c_star)
                    input_src = vcat(input_src, rho_ti_c_star)
                    input_src = vcat(input_src, c_ti_c_star)
                end
            end
            if settings.airframe
                input_src = vcat(input_src, theta_flaps)
                input_src = vcat(input_src, I_landing_gear)
            end
            if settings.fan_inlet==true || settings.fan_discharge==true
                input_src = vcat(input_src, DTt_f_star)
                input_src = vcat(input_src, mdot_f_star)
                input_src = vcat(input_src, N_f_star)
                input_src = vcat(input_src, A_f_star)
                input_src = vcat(input_src, d_f_star)
            end

            if settings.shielding == true
                msap_source[i,:,:] = source(settings, data, ac, shield[i,:,:], n_t, idx_src, input_src)
            else
                msap_source[i,:,:] = source(settings, data, ac, zeros(T, (n_t, settings.N_f)), n_t, idx_src, input_src)
            end

            # Compute propagation
            input_prop = r
            input_prop = vcat(input_prop, x)
            input_prop = vcat(input_prop, z)
            input_prop = vcat(input_prop, c_bar)
            input_prop = vcat(input_prop, rho_0)
            input_prop = vcat(input_prop, I_0)
            input_prop = vcat(input_prop, beta)
            msap_prop[i,:,:] = propagation(settings, data, x_obs, n_t, msap_source[i,:,:], input_prop)

            # Compute Levels
            spl[i,:,:] = f_spl(settings, msap_prop[i,:,:], rho_0, c_0)
            oaspl[i,:] = f_oaspl(settings, spl[i,:,:])
            pnlt[i,:], C[i,:,:] = f_pnlt(settings, data, n_t, spl[i,:,:])

            # Compute integrated levels
            if settings.levels_int_metric == "ioaspl"
                level_int[i] = f_ioaspl(settings, n_t, oaspl[i,:], t_o[i,:])
            elseif settings.levels_int_metric == "ipnlt"
                level_int[i] = f_ipnlt(settings, n_t, pnlt[i,:], t_o[i,:])
            elseif settings.levels_int_metric == "epnl"
                level_int[i] = f_epnl(settings, n_t, pnlt[i,:], t_o[i,:])
            end
        end

        # Write output
        if optimization == true
            level_opt = maximum(level_int[1:end-1]) + level_int[end]
            return level_opt
        else
            return t_o, msap_source, msap_prop, spl, oaspl, pnlt, level_int
        end
    end

    # Default values for input vector
    X = range(1, 10000, length=n_t)             # x
    X = vcat(X, zeros(Float64, n_t))            # y
    X = vcat(X, range(1, 1000, length=n_t))     # z
    X = vcat(X, 10. * ones(Float64, n_t))       # alpha
    X = vcat(X, 10. * ones(Float64, n_t))       # gamma
    X = vcat(X, range(0, 100, length=n_t))      # t_s
    X = vcat(X, 1.225 * ones(Float64, n_t))     # rho_0
    X = vcat(X, 1.789e-5 * ones(Float64, n_t))  # mu_0
    X = vcat(X, 340.294 * ones(Float64, n_t))   # c_0
    X = vcat(X, 288.15 * ones(Float64, n_t))    # T_0
    X = vcat(X, 101325. * ones(Float64, n_t))   # p_0
    X = vcat(X, 0.3 * ones(Float64, n_t))       # M_0
    X = vcat(X, 400. * ones(Float64, n_t))      # I_0
    X = vcat(X, 1. * ones(Float64, n_t))        # TS
    if settings.jet_mixing == true && settings.jet_shock == false
        X = vcat(X, 400. * ones(Float64, n_t))  # V_j
        X = vcat(X, 0.8  * ones(Float64, n_t))  # rho_j
        X = vcat(X, 0.5  * ones(Float64, n_t))  # A_j
        X = vcat(X, 500. * ones(Float64, n_t))  # Tt_j
    elseif settings.jet_shock == true && settings.jet_mixing == false
        X = vcat(X, 400. * ones(Float64, n_t))  # V_j
        X = vcat(X, 0.5  * ones(Float64, n_t))  # A_j
        X = vcat(X, 500. * ones(Float64, n_t))  # Tt_j
        X = vcat(X, 1.   * ones(Float64, n_t))  # M_j
    elseif settings.jet_shock ==true && settings.jet_mixing == true
        X = vcat(X, 400. * ones(Float64, n_t))  # V_j
        X = vcat(X, 0.8  * ones(Float64, n_t))  # rho_j
        X = vcat(X, 0.5  * ones(Float64, n_t))  # A_j
        X = vcat(X, 500. * ones(Float64, n_t))  # Tt_j
        X = vcat(X, 1.   * ones(Float64, n_t))  # M_j
    end
    if settings.core
        if settings.method_core_turb == "GE"
            X = vcat(X, 30.   * ones(Float64, n_t))    # mdot_c
            X = vcat(X, 20.e6 * ones(Float64, n_t))  # Tti_c
            X = vcat(X, 800.  *  ones(Float64, n_t))  # Ttj_c
            X = vcat(X, 1600. * ones(Float64, n_t))  # Pti_c
            X = vcat(X, 800.  * ones(Float64, n_t))  # DTt_des_c
        elseif settings.method_core_turb == "PW"
            X = vcat(X, 30.   * ones(Float64, n_t))  # mdot_c
            X = vcat(X, 20e6  * ones(Float64, n_t))  # Tti_c
            X = vcat(X, 800.  * ones(Float64, n_t))  # Ttj_c
            X = vcat(X, 1600. * ones(Float64, n_t))  # Pti_c
            X = vcat(X, 0.5   * ones(Float64, n_t))  # rho_te_c
            X = vcat(X, 600.  * ones(Float64, n_t))  # c_te_c
            X = vcat(X, 3.5   * ones(Float64, n_t))  # rho_ti_c
            X = vcat(X, 800.  * ones(Float64, n_t))  # c_ti_c
        end
    end
    if settings.airframe
        X = vcat(X, 10. * ones(Float64, n_t)) # theta_flap
        X = vcat(X, ones(Float64, n_t))       # I_landing_gear
    end
    if settings.fan_inlet==true || settings.fan_discharge==true
        X = vcat(X, 70.   * ones(Float64, n_t))
        X = vcat(X, 200.  * ones(Float64, n_t))
        X = vcat(X, 8000. * ones(Float64, n_t))
        X = vcat(X, 0.9   * ones(Float64, n_t))
        X = vcat(X, 1.7   * ones(Float64, n_t))
    end

    # Default values for output value and jacobian
    Y = 100.
    J = Y.*X'
    #'

    # Define noise function
    noise_fwd = (x)->noise(settings, data, ac, n_t, idx, idx_src, optimization, x)

    # Create compiled ReverseDiff tape 
    # if optimization == true
    #     noise_tape = ReverseDiff.GradientTape(noise_fwd, X)
    # else
    noise_tape = 0        
    # end

    return Noise(settings, data, ac, n_t, idx, idx_src, optimization, noise, X, J, noise_fwd, noise_tape)
end

function setup(self::Noise)
    # Load options
    settings = self.settings
    n_t = self.n_t

    # Number of observers
    n_obs = size(settings.x_observer_array)[1]

    # Define inputs --------------------------------------------------------------------------------
    inputs = Vector{VarData}()
    push!(inputs, VarData("x", shape=(n_t, ), val=ones(n_t), units="m"))
    push!(inputs, VarData("y", shape=(n_t, ), val=ones(n_t), units="m"))
    push!(inputs, VarData("z", shape=(n_t, ), val=ones(n_t), units="m"))
    push!(inputs, VarData("alpha", shape=(n_t, ), val=ones(n_t), units="deg"))
    push!(inputs, VarData("gamma", shape=(n_t, ), val=ones(n_t), units="deg"))
    push!(inputs, VarData("t_s", shape=(n_t, ), val=ones(n_t), units="s"))
    push!(inputs, VarData("rho_0", shape=(n_t,), val=ones(n_t), units="kg/m**3"))
    push!(inputs, VarData("mu_0", shape=(n_t,), val=ones(n_t), units="kg/m/s"))
    push!(inputs, VarData("c_0", shape=(n_t, ), val=ones(n_t), units="m/s"))
    push!(inputs, VarData("T_0", shape=(n_t, ), val=ones(n_t), units="K"))
    push!(inputs, VarData("p_0", shape=(n_t, ), val=ones(n_t), units="Pa"))
    push!(inputs, VarData("M_0", shape=(n_t,), val=ones(n_t)))
    push!(inputs, VarData("I_0", shape=(n_t,), val=ones(n_t), units="kg/m**2/s"))
    push!(inputs, VarData("TS", shape=(n_t,), val=ones(n_t)))

    if settings.jet_mixing == true && settings.jet_shock == false
        push!(inputs, VarData("V_j", shape=(n_t,), val=ones(n_t), units="m/s"))
        push!(inputs, VarData("rho_j", shape=(n_t,), val=ones(n_t), units="kg/m**3"))
        push!(inputs, VarData("A_j", shape=(n_t,), val=ones(n_t), units="m**2"))
        push!(inputs, VarData("Tt_j", shape=(n_t,), val=ones(n_t), units="K"))
    elseif settings.jet_shock == true && settings.jet_mixing == false
        push!(inputs, VarData("V_j", shape=(n_t,), val=ones(n_t), units="m/s"))
        push!(inputs, VarData("M_j", shape=(n_t,), val=ones(n_t)))
        push!(inputs, VarData("A_j", shape=(n_t,), val=ones(n_t), units="m**2"))
        push!(inputs, VarData("Tt_j", shape=(n_t,), val=ones(n_t), units="K"))
    elseif settings.jet_shock ==true && settings.jet_mixing == true
        push!(inputs, VarData("V_j", shape=(n_t,), val=ones(n_t), units="m/s"))
        push!(inputs, VarData("rho_j", shape=(n_t,), val=ones(n_t), units="kg/m**3"))
        push!(inputs, VarData("A_j", shape=(n_t,), val=ones(n_t), units="m**2"))
        push!(inputs, VarData("Tt_j", shape=(n_t,), val=ones(n_t), units="K"))
        push!(inputs, VarData("M_j", shape=(n_t,), val=ones(n_t)))
    end
    if settings.core
        if settings.method_core_turb == "GE"
            push!(inputs, VarData("mdoti_c", shape=(n_t,), val=ones(n_t), units="kg/s"))
            push!(inputs, VarData("Tti_c", shape=(n_t,), val=ones(n_t), units="K"))
            push!(inputs, VarData("Ttj_c", shape=(n_t,), val=ones(n_t), units="K"))
            push!(inputs, VarData("Pti_c", shape=(n_t,), val=ones(n_t), units="Pa"))
            push!(inputs, VarData("DTt_des_c", shape=(n_t,), val=ones(n_t), units="K"))
        elseif settings.method_core_turb == "PW"
            push!(inputs, VarData("mdoti_c", shape=(n_t,), val=ones(n_t), units="kg/s"))
            push!(inputs, VarData("Tti_c", shape=(n_t,), val=ones(n_t), units="K"))
            push!(inputs, VarData("Ttj_c", shape=(n_t,), val=ones(n_t), units="K"))
            push!(inputs, VarData("Pti_c", shape=(n_t,), val=ones(n_t), units="Pa"))
            push!(inputs, VarData("rho_te_c", shape=(n_t,), val=ones(n_t), units="kg/m**3"))
            push!(inputs, VarData("c_te_c", shape=(n_t,), val=ones(n_t), units="m/s"))
            push!(inputs, VarData("rho_ti_c", shape=(n_t,), val=ones(n_t), units="kg/m**3"))
            push!(inputs, VarData("c_ti_c", shape=(n_t,), val=ones(n_t), units="m/s"))
        end
    end
    if settings.airframe
        push!(inputs, VarData("theta_flaps", shape=(n_t,), val=ones(n_t), units="deg"))
        push!(inputs, VarData("I_landing_gear", shape=(n_t,), val=ones(n_t)))
    end
    if settings.fan_inlet==true || settings.fan_discharge==true
        push!(inputs, VarData("DTt_f", shape=(n_t,), val=ones(n_t), units="K"))
        push!(inputs, VarData("mdot_f", shape=(n_t,), val=ones(n_t), units="kg/s"))
        push!(inputs, VarData("N_f", shape=(n_t,), val=ones(n_t), units="rpm"))
        push!(inputs, VarData("A_f", shape=(n_t,), val=ones(n_t), units="m**2"))
        push!(inputs, VarData("d_f", shape=(n_t,), val=ones(n_t), units="m"))
    end

    # Define outputs --------------------------------------------------------------------------------
    outputs = Vector{VarData}()
    if self.optimization == true
        push!(outputs, VarData(settings.levels_int_metric, shape=(1, ), val=0.))
    else
        push!(outputs, VarData("t_o", shape=(n_obs, n_t), val=ones(n_obs, n_t)))
        push!(outputs, VarData("msap_source", shape=(n_obs, n_t, settings.N_f), val=ones(n_obs, n_t, settings.N_f)))
        push!(outputs, VarData("msap_prop", shape=(n_obs, n_t, settings.N_f), val=ones(n_obs, n_t, settings.N_f)))
        push!(outputs, VarData("spl", shape=(n_obs, n_t, settings.N_f), val=ones(n_obs, n_t, settings.N_f)))
        push!(outputs, VarData("oaspl", shape=(n_obs, n_t), val=ones(n_obs, n_t)))
        push!(outputs, VarData("pnlt", shape=(n_obs, n_t), val=ones(n_obs, n_t)))
        push!(outputs, VarData(settings.levels_int_metric, shape=(n_obs, ), val=ones(n_obs, )))
    end

    ## Define partials --------------------------------------------------------------------------------
    partials = Vector{PartialsData}()
    
    push!(partials, PartialsData(settings.levels_int_metric, "x"))
    push!(partials, PartialsData(settings.levels_int_metric, "y"))
    push!(partials, PartialsData(settings.levels_int_metric, "z"))
    push!(partials, PartialsData(settings.levels_int_metric, "alpha"))
    push!(partials, PartialsData(settings.levels_int_metric, "gamma"))
    push!(partials, PartialsData(settings.levels_int_metric, "t_s"))
    push!(partials, PartialsData(settings.levels_int_metric, "rho_0"))
    push!(partials, PartialsData(settings.levels_int_metric, "mu_0"))
    push!(partials, PartialsData(settings.levels_int_metric, "c_0"))
    push!(partials, PartialsData(settings.levels_int_metric, "T_0"))
    push!(partials, PartialsData(settings.levels_int_metric, "p_0"))
    push!(partials, PartialsData(settings.levels_int_metric, "M_0"))
    push!(partials, PartialsData(settings.levels_int_metric, "I_0"))
    push!(partials, PartialsData(settings.levels_int_metric, "TS"))
    
    if settings.jet_mixing == true && settings.jet_shock == false
        push!(partials, PartialsData(settings.levels_int_metric, "V_j"))
        push!(partials, PartialsData(settings.levels_int_metric, "rho_j"))
        push!(partials, PartialsData(settings.levels_int_metric, "A_j"))
        push!(partials, PartialsData(settings.levels_int_metric, "Tt_j"))
    elseif settings.jet_shock == true && settings.jet_mixing == false
        push!(partials, PartialsData(settings.levels_int_metric, "V_j"))
        push!(partials, PartialsData(settings.levels_int_metric, "M_j"))
        push!(partials, PartialsData(settings.levels_int_metric, "A_j"))
        push!(partials, PartialsData(settings.levels_int_metric, "Tt_j"))
    elseif settings.jet_shock ==true && settings.jet_mixing == true
        push!(partials, PartialsData(settings.levels_int_metric, "V_j"))
        push!(partials, PartialsData(settings.levels_int_metric, "rho_j"))
        push!(partials, PartialsData(settings.levels_int_metric, "A_j"))
        push!(partials, PartialsData(settings.levels_int_metric, "Tt_j"))
        push!(partials, PartialsData(settings.levels_int_metric, "M_j"))
    end
    if settings.core
        if settings.method_core_turb == "GE"
            push!(partials, PartialsData(settings.levels_int_metric, "mdoti_c"))
            push!(partials, PartialsData(settings.levels_int_metric, "Tti_c"))
            push!(partials, PartialsData(settings.levels_int_metric, "Ttj_c"))
            push!(partials, PartialsData(settings.levels_int_metric, "Pti_c"))
            push!(partials, PartialsData(settings.levels_int_metric, "DTt_des_c"))
        elseif settings.method_core_turb == "PW"
            push!(partials, PartialsData(settings.levels_int_metric, "mdoti_c"))
            push!(partials, PartialsData(settings.levels_int_metric, "Tti_c"))
            push!(partials, PartialsData(settings.levels_int_metric, "Ttj_c"))
            push!(partials, PartialsData(settings.levels_int_metric, "Pti_c"))
            push!(partials, PartialsData(settings.levels_int_metric, "rho_te_c"))
            push!(partials, PartialsData(settings.levels_int_metric, "c_te_c"))
            push!(partials, PartialsData(settings.levels_int_metric, "rho_ti_c"))
            push!(partials, PartialsData(settings.levels_int_metric, "c_ti_c"))
        end
    end
    if settings.airframe
        push!(partials, PartialsData(settings.levels_int_metric, "theta_flaps"))
        push!(partials, PartialsData(settings.levels_int_metric, "I_landing_gear"))
    end
    if settings.fan_inlet==true || settings.fan_discharge==true
        push!(partials, PartialsData(settings.levels_int_metric, "DTt_f"))
        push!(partials, PartialsData(settings.levels_int_metric, "mdot_f"))
        push!(partials, PartialsData(settings.levels_int_metric, "N_f"))
        push!(partials, PartialsData(settings.levels_int_metric, "A_f"))
        push!(partials, PartialsData(settings.levels_int_metric, "d_f"))
    end

    return inputs, outputs, partials
end

function compute!(self::Noise, inputs, outputs)
    # Load options
    settings = self.settings
    data = self.data
    ac = self.ac
    n_t = self.n_t
    idx = self.idx
    idx_src = self.idx_src
    X = self.X

    ## Print inputs to file
    #open(string(Dates.today())*"-inputs_TS.txt","a") do io
    #    println(io,inputs["TS"])
    #end

    # Extract inputs
    X[idx["x"][1]:idx["x"][2]] = inputs["x"]
    X[idx["y"][1]:idx["y"][2]] = inputs["y"]
    X[idx["z"][1]:idx["z"][2]] = inputs["z"]
    X[idx["alpha"][1]:idx["alpha"][2]] = inputs["alpha"]
    X[idx["gamma"][1]:idx["gamma"][2]] = inputs["gamma"]
    X[idx["t_s"][1]:idx["t_s"][2]] = inputs["t_s"]
    X[idx["rho_0"][1]:idx["rho_0"][2]] = inputs["rho_0"]
    X[idx["mu_0"][1]:idx["mu_0"][2]] = inputs["mu_0"]
    X[idx["c_0"][1]:idx["c_0"][2]] = inputs["c_0"]
    X[idx["T_0"][1]:idx["T_0"][2]] = inputs["T_0"]
    X[idx["p_0"][1]:idx["p_0"][2]] = inputs["p_0"]
    X[idx["M_0"][1]:idx["M_0"][2]] = inputs["M_0"]
    X[idx["I_0"][1]:idx["I_0"][2]] = inputs["I_0"]
    X[idx["TS"][1]:idx["TS"][2]] = inputs["TS"]
    
    if settings.jet_mixing == true && settings.jet_shock == false
        X[idx["V_j"][1]:idx["V_j"][2]] = inputs["V_j"]
        X[idx["rho_j"][1]:idx["rho_j"][2]] = inputs["rho_j"]
        X[idx["A_j"][1]:idx["A_j"][2]] = inputs["A_j"]
        X[idx["Tt_j"][1]:idx["Tt_j"][2]] = inputs["Tt_j"]
    elseif settings.jet_shock == true && settings.jet_mixing == false
        X[idx["V_j"][1]:idx["V_j"][2]] = inputs["V_j"]
        X[idx["M_j"][1]:idx["M_j"][2]] = inputs["M_j"]
        X[idx["A_j"][1]:idx["A_j"][2]] = inputs["A_j"]
        X[idx["Tt_j"][1]:idx["Tt_j"][2]] = inputs["Tt_j"]
    elseif settings.jet_shock ==true && settings.jet_mixing == true
        X[idx["V_j"][1]:idx["V_j"][2]] = inputs["V_j"]
        X[idx["rho_j"][1]:idx["rho_j"][2]] = inputs["rho_j"]
        X[idx["A_j"][1]:idx["A_j"][2]] = inputs["A_j"]
        X[idx["Tt_j"][1]:idx["Tt_j"][2]] = inputs["Tt_j"]
        X[idx["M_j"][1]:idx["M_j"][2]] = inputs["M_j"]
    end
    if settings.core
        if settings.method_core_turb == "GE"
            X[idx["mdoti_c"][1]:idx["mdoti_c"][2]] = inputs["mdoti_c"]
            X[idx["Tti_c"][1]:idx["Tti_c"][2]] = inputs["Tti_c"]
            X[idx["Ttj_c"][1]:idx["Ttj_c"][2]] = inputs["Ttj_c"]
            X[idx["Pti_c"][1]:idx["Pti_c"][2]] = inputs["Pti_c"]
            X[idx["DTt_des_c"][1]:idx["DTt_des_c"][2]] = inputs["DTt_des_c"]
        elseif settings.method_core_turb == "PW"
            X[idx["mdoti_c"][1]:idx["mdoti_c"][2]] = inputs["mdoti_c"]
            X[idx["Tti_c"][1]:idx["Tti_c"][2]] = inputs["Tti_c"]
            X[idx["Ttj_c"][1]:idx["Ttj_c"][2]] = inputs["Ttj_c"]
            X[idx["Pti_c"][1]:idx["Pti_c"][2]] = inputs["Pti_c"]
            X[idx["rho_te_c"][1]:idx["rho_te_c"][2]] = inputs["rho_te_c"]
            X[idx["c_te_c"][1]:idx["c_te_c"][2]] = inputs["c_te_c"]
            X[idx["rho_ti_c"][1]:idx["rho_ti_c"][2]] = inputs["rho_ti_c"]
            X[idx["c_ti_c"][1]:idx["c_ti_c"][2]] = inputs["c_ti_c"]
        end
    end
    if settings.airframe
        X[idx["theta_flaps"][1]:idx["theta_flaps"][2]] = inputs["theta_flaps"]
        X[idx["I_landing_gear"][1]:idx["I_landing_gear"][2]] = inputs["I_landing_gear"]
    end
    if settings.fan_inlet==true || settings.fan_discharge==true
        X[idx["DTt_f"][1]:idx["DTt_f"][2]] = inputs["DTt_f"]
        X[idx["mdot_f"][1]:idx["mdot_f"][2]] = inputs["mdot_f"]
        X[idx["N_f"][1]:idx["N_f"][2]] = inputs["N_f"]
        X[idx["A_f"][1]:idx["A_f"][2]] = inputs["A_f"]
        X[idx["d_f"][1]:idx["d_f"][2]] = inputs["d_f"]
    end

    if self.optimization == true
        levels_int = self.noise(settings, data, ac, n_t, idx, idx_src, self.optimization, X)
        @. outputs[settings.levels_int_metric] = levels_int

        # Print outputs to file
        open(string(Dates.today())*"-outputs_levels_int.txt", "a") do io
            println(io, levels_int)
        end
    
    else
        t_o, msap_source, msap_prop, spl, oaspl, pnlt, levels_int = self.noise(settings, data, ac, n_t, idx, idx_src, self.optimization, X)
        @. outputs["t_o"] = t_o
        @. outputs["msap_source"] = msap_source
        @. outputs["msap_prop"] = msap_prop
        @. outputs["spl"] = spl
        @. outputs["oaspl"] = oaspl
        @. outputs["pnlt"] = pnlt
        @. outputs[settings.levels_int_metric] = levels_int
    end

end

function compute_partials!(self::Noise, inputs, partials)
    # Load options
    settings = self.settings
    n_t = self.n_t
    idx = self.idx
    X = self.X
    J = self.J

    # Print start statement
    println("Computing partials noise")

    # Extract inputs
    X[idx["x"][1]:idx["x"][2]] = inputs["x"]
    X[idx["y"][1]:idx["y"][2]] = inputs["y"]
    X[idx["z"][1]:idx["z"][2]] = inputs["z"]
    X[idx["alpha"][1]:idx["alpha"][2]] = inputs["alpha"]
    X[idx["gamma"][1]:idx["gamma"][2]] = inputs["gamma"]
    X[idx["t_s"][1]:idx["t_s"][2]] = inputs["t_s"]
    X[idx["rho_0"][1]:idx["rho_0"][2]] = inputs["rho_0"]
    X[idx["mu_0"][1]:idx["mu_0"][2]] = inputs["mu_0"]
    X[idx["c_0"][1]:idx["c_0"][2]] = inputs["c_0"]
    X[idx["T_0"][1]:idx["T_0"][2]] = inputs["T_0"]
    X[idx["p_0"][1]:idx["p_0"][2]] = inputs["p_0"]
    X[idx["M_0"][1]:idx["M_0"][2]] = inputs["M_0"]
    X[idx["I_0"][1]:idx["I_0"][2]] = inputs["I_0"]
    X[idx["TS"][1]:idx["TS"][2]] = inputs["TS"]
    
    if settings.jet_mixing == true && settings.jet_shock == false
        X[idx["V_j"][1]:idx["V_j"][2]] = inputs["V_j"]
        X[idx["rho_j"][1]:idx["rho_j"][2]] = inputs["rho_j"]
        X[idx["A_j"][1]:idx["A_j"][2]] = inputs["A_j"]
        X[idx["Tt_j"][1]:idx["Tt_j"][2]] = inputs["Tt_j"]
    elseif settings.jet_shock == true && settings.jet_mixing == false
        X[idx["V_j"][1]:idx["V_j"][2]] = inputs["V_j"]
        X[idx["M_j"][1]:idx["M_j"][2]] = inputs["M_j"]
        X[idx["A_j"][1]:idx["A_j"][2]] = inputs["A_j"]
        X[idx["Tt_j"][1]:idx["Tt_j"][2]] = inputs["Tt_j"]
    elseif settings.jet_shock ==true && settings.jet_mixing == true
        X[idx["V_j"][1]:idx["V_j"][2]] = inputs["V_j"]
        X[idx["rho_j"][1]:idx["rho_j"][2]] = inputs["rho_j"]
        X[idx["A_j"][1]:idx["A_j"][2]] = inputs["A_j"]
        X[idx["Tt_j"][1]:idx["Tt_j"][2]] = inputs["Tt_j"]
        X[idx["M_j"][1]:idx["M_j"][2]] = inputs["M_j"]
    end
    if settings.core
        if settings.method_core_turb == "GE"
            X[idx["mdoti_c"][1]:idx["mdoti_c"][2]] = inputs["mdoti_c"]
            X[idx["Tti_c"][1]:idx["Tti_c"][2]] = inputs["Tti_c"]
            X[idx["Ttj_c"][1]:idx["Ttj_c"][2]] = inputs["Ttj_c"]
            X[idx["Pti_c"][1]:idx["Pti_c"][2]] = inputs["Pti_c"]
            X[idx["DTt_des_c"][1]:idx["DTt_des_c"][2]] = inputs["DTt_des_c"]
        elseif settings.method_core_turb == "PW"
            X[idx["mdoti_c"][1]:idx["mdoti_c"][2]] = inputs["mdoti_c"]
            X[idx["Tti_c"][1]:idx["Tti_c"][2]] = inputs["Tti_c"]
            X[idx["Ttj_c"][1]:idx["Ttj_c"][2]] = inputs["Ttj_c"]
            X[idx["Pti_c"][1]:idx["Pti_c"][2]] = inputs["Pti_c"]
            X[idx["rho_te_c"][1]:idx["rho_te_c"][2]] = inputs["rho_te_c"]
            X[idx["c_te_c"][1]:idx["c_te_c"][2]] = inputs["c_te_c"]
            X[idx["rho_ti_c"][1]:idx["rho_ti_c"][2]] = inputs["rho_ti_c"]
            X[idx["c_ti_c"][1]:idx["c_ti_c"][2]] = inputs["c_ti_c"]
        end
    end
    if settings.airframe
        X[idx["theta_flaps"][1]:idx["theta_flaps"][2]] = inputs["theta_flaps"]
        X[idx["I_landing_gear"][1]:idx["I_landing_gear"][2]] = inputs["I_landing_gear"]
    end
    if settings.fan_inlet==true || settings.fan_discharge==true
        X[idx["DTt_f"][1]:idx["DTt_f"][2]] = inputs["DTt_f"]
        X[idx["mdot_f"][1]:idx["mdot_f"][2]] = inputs["mdot_f"]
        X[idx["N_f"][1]:idx["N_f"][2]] = inputs["N_f"]
        X[idx["A_f"][1]:idx["A_f"][2]] = inputs["A_f"]
        X[idx["d_f"][1]:idx["d_f"][2]] = inputs["d_f"]
    end

    # compute Jacobian using compiled tape
    ReverseDiff.gradient!(J, self.noise_fwd, X)
    
    # println(J)
    # k = lala

    dnoise_dx = reshape(J[idx["x"][1]:idx["x"][2]], (1, n_t))
    @. partials[settings.levels_int_metric, "x"] = dnoise_dx
    dnoise_dy = reshape(J[idx["y"][1]:idx["y"][2]], (1, n_t))
    @. partials[settings.levels_int_metric, "y"] = dnoise_dy
    dnoise_dz = reshape(J[idx["z"][1]:idx["z"][2]], (1, n_t))
    @. partials[settings.levels_int_metric, "z"] = dnoise_dz
    dnoise_dalpha = reshape(J[idx["alpha"][1]:idx["alpha"][2]], (1, n_t))
    @. partials[settings.levels_int_metric, "alpha"] = dnoise_dalpha
    dnoise_dgamma = reshape(J[idx["gamma"][1]:idx["gamma"][2]], (1, n_t))
    @. partials[settings.levels_int_metric, "gamma"] = dnoise_dgamma
    dnoise_dts = reshape(J[idx["t_s"][1]:idx["t_s"][2]], (1, n_t))
    @. partials[settings.levels_int_metric, "t_s"] = dnoise_dts
    dnoise_drho_0 = reshape(J[idx["rho_0"][1]:idx["rho_0"][2]], (1, n_t))
    @. partials[settings.levels_int_metric, "rho_0"] = dnoise_drho_0
    dnoise_dmu_0 = reshape(J[idx["mu_0"][1]:idx["mu_0"][2]], (1, n_t))
    @. partials[settings.levels_int_metric, "mu_0"] = dnoise_dmu_0
    dnoise_dc_0 = reshape(J[idx["c_0"][1]:idx["c_0"][2]], (1, n_t))
    @. partials[settings.levels_int_metric, "c_0"] = dnoise_dc_0
    dnoise_dT_0 = reshape(J[idx["T_0"][1]:idx["T_0"][2]], (1, n_t))
    @. partials[settings.levels_int_metric, "T_0"] = dnoise_dT_0
    dnoise_dp_0 = reshape(J[idx["p_0"][1]:idx["p_0"][2]], (1, n_t))
    @. partials[settings.levels_int_metric, "p_0"] = dnoise_dp_0
    dnoise_dM_0 = reshape(J[idx["M_0"][1]:idx["M_0"][2]], (1, n_t))
    @. partials[settings.levels_int_metric, "M_0"] = dnoise_dM_0
    dnoise_dI_0 = reshape(J[idx["I_0"][1]:idx["I_0"][2]], (1, n_t))
    @. partials[settings.levels_int_metric, "I_0"] = dnoise_dI_0
    dnoise_dTS = reshape(J[idx["TS"][1]:idx["TS"][2]], (1, n_t))
    @. partials[settings.levels_int_metric, "TS"] = dnoise_dTS

    if settings.jet_mixing == true && settings.jet_shock == false
        dnoise_dvj = reshape(J[idx["V_j"][1]:idx["V_j"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "V_j"] = dnoise_dvj
        dnoise_drhoj = reshape(J[idx["rho_j"][1]:idx["rho_j"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "rho_j"] = dnoise_drhoj
        dnoise_dAj = reshape(J[idx["A_j"][1]:idx["A_j"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "A_j"] = dnoise_dAj
        dnoise_dTtj = reshape(J[idx["Tt_j"][1]:idx["Tt_j"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "Tt_j"] = dnoise_dTtj
    elseif settings.jet_shock == true && settings.jet_mixing == false
        dnoise_dvj = reshape(J[idx["V_j"][1]:idx["V_j"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "V_j"] = dnoise_dvj
        dnoise_dMj = reshape(J[idx["M_j"][1]:idx["M_j"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "M_j"] = dnoise_dMj
        dnoise_dAj = reshape(J[idx["A_j"][1]:idx["A_j"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "A_j"] = dnoise_dAj
        dnoise_dTtj = reshape(J[idx["Tt_j"][1]:idx["Tt_j"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "Tt_j"] = dnoise_dTtj
    elseif settings.jet_shock ==true && settings.jet_mixing == true
        dnoise_dvj = reshape(J[idx["V_j"][1]:idx["V_j"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "V_j"] = dnoise_dvj
        dnoise_drhoj = reshape(J[idx["rho_j"][1]:idx["rho_j"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "rho_j"] = dnoise_drhoj
        dnoise_dAj = reshape(J[idx["A_j"][1]:idx["A_j"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "A_j"] = dnoise_dAj
        dnoise_dTtj = reshape(J[idx["Tt_j"][1]:idx["Tt_j"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "Tt_j"] = dnoise_dTtj
        dnoise_dMj = reshape(J[idx["M_j"][1]:idx["M_j"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "M_j"] = dnoise_dMj
    end
    if settings.core
        if settings.method_core_turb == "GE"
            dnoise_dmdoti_c = reshape(J[idx["mdoti_c"][1]:idx["mdoti_c"][2]], (1, n_t))
            @. partials[settings.levels_int_metric, "mdoti_c"] = dnoise_dmdoti_c
            dnoise_dTti_c = reshape(J[idx["Tti_c"][1]:idx["Tti_c"][2]], (1, n_t))
            @. partials[settings.levels_int_metric, "Tti_c"] = dnoise_dTti_c
            dnoise_dTtj_c = reshape(J[idx["Ttj_c"][1]:idx["Ttj_c"][2]], (1, n_t))
            @. partials[settings.levels_int_metric, "Ttj_c"] = dnoise_dTtj_c
            dnoise_dPti_c = reshape(J[idx["Pti_c"][1]:idx["Pti_c"][2]], (1, n_t))
            @. partials[settings.levels_int_metric, "Pti_c"] = dnoise_dPti_c
            dnoise_dDTt_des_c = reshape(J[idx["DTt_des_c"][1]:idx["DTt_des_c"][2]], (1, n_t))
            @. partials[settings.levels_int_metric, "DTt_des_c"] = dnoise_dDTt_des_c
        elseif settings.method_core_turb == "PW"
            dnoise_dmdoti_c = reshape(J[idx["mdoti_c"][1]:idx["mdoti_c"][2]], (1, n_t))
            @. partials[settings.levels_int_metric, "mdoti_c"] = dnoise_dmdoti_c
            dnoise_dTti_c = reshape(J[idx["Tti_c"][1]:idx["Tti_c"][2]], (1, n_t))
            @. partials[settings.levels_int_metric, "Tti_c"] = dnoise_dTti_c
            dnoise_dTtj_c = reshape(J[idx["Ttj_c"][1]:idx["Ttj_c"][2]], (1, n_t))
            @. partials[settings.levels_int_metric, "Ttj_c"] = dnoise_dTtj_c
            dnoise_dPti_c = reshape(J[idx["Pti_c"][1]:idx["Pti_c"][2]], (1, n_t))
            @. partials[settings.levels_int_metric, "Pti_c"] = dnoise_dPti_c
            dnoise_drho_te_c = reshape(J[idx["rho_te_c"][1]:idx["rho_te_c"][2]], (1, n_t))
            @. partials[settings.levels_int_metric, "rho_te_c"] = dnoise_drho_te_c
            dnoise_dc_te_c = reshape(J[idx["c_te_c"][1]:idx["c_te_c"][2]], (1, n_t))
            @. partials[settings.levels_int_metric, "c_te_c"] = dnoise_dc_te_c
            dnoise_drho_ti_c = reshape(J[idx["rho_ti_c"][1]:idx["rho_ti_c"][2]], (1, n_t))
            @. partials[settings.levels_int_metric, "rho_ti_c"] = dnoise_drho_ti_c
            dnoise_dc_ti_c = reshape(J[idx["c_ti_c"][1]:idx["c_ti_c"][2]], (1, n_t))
            @. partials[settings.levels_int_metric, "c_ti_c"] = dnoise_dc_ti_c
        end
    end
    if settings.airframe
        dnoise_dtheta_flaps = reshape(J[idx["theta_flaps"][1]:idx["theta_flaps"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "theta_flaps"] = dnoise_dtheta_flaps
        dnoise_dI_landing_gear = reshape(J[idx["I_landing_gear"][1]:idx["I_landing_gear"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "I_landing_gear"] = dnoise_dI_landing_gear
    end

    if settings.fan_inlet==true || settings.fan_discharge==true
        dnoise_dDTt_f = reshape(J[idx["DTt_f"][1]:idx["DTt_f"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "DTt_f"] = dnoise_dDTt_f
        dnoise_dmdot_f = reshape(J[idx["mdot_f"][1]:idx["mdot_f"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "mdot_f"] = dnoise_dmdot_f
        dnoise_dN_f = reshape(J[idx["N_f"][1]:idx["N_f"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "N_f"] = dnoise_dN_f
        dnoise_dA_f = reshape(J[idx["A_f"][1]:idx["A_f"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "A_f"] = dnoise_dA_f
        dnoise_dd_f = reshape(J[idx["d_f"][1]:idx["d_f"][2]], (1, n_t))
        @. partials[settings.levels_int_metric, "d_f"] = dnoise_dd_f
    end

    println("Done computing partials noise.")

end