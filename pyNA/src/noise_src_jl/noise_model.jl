using OpenMDAO: AbstractExplicitComp, VarData, PartialsData, get_rows_cols
import OpenMDAO: setup, compute!, compute_partials!
using Interpolations
using ReverseDiff
using LinearAlgebra
using ConcreteStructs
using PCHIPInterpolation
using Dates
using FLoops
using BenchmarkTools

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
include("aspl.jl")
include("sel.jl")
include("smooth_max.jl")


# Define propagation struct
@concrete struct NoiseModel <: AbstractExplicitComp
    settings :: PyObject
    data :: PyObject
    ac :: PyObject
    n_t :: Int64
    idx :: Dict{Any, Any}
    objective :: String
    noise_model
    X :: Array{Float64, 1}
    J :: Array{Float64, 2}
    noise_model_fwd
end

function NoiseModel(settings::PyObject, data::PyObject, ac::PyObject, n_t::Int64, idx::Dict{Any, Any}, objective::String)

    function compute_noise_observer(settings::PyObject, data::PyObject, ac::PyObject, n_t::Int64, idx::Dict{Any, Any}, x_obs::Array{Float64, 1}, shield::Array{Float64, 2}, idx_supp::Array{CartesianIndex{2}, 1}, objective::String, input_v::Union{Array{Float64, 1}, ReverseDiff.TrackedArray{Float64,Float64,1,Array{Float64,1},Array{Float64,1}}})

        # Initialize
        spl = zeros(eltype(input_v), (n_t, settings.N_f))

        # Compute geometry
        r, beta, theta, phi, c_bar, t_o = geometry(settings, x_obs, input_v[idx["x"][1]:idx["x"][2]], input_v[idx["y"][1]:idx["y"][2]], input_v[idx["z"][1]:idx["z"][2]], input_v[idx["alpha"][1]:idx["alpha"][2]], input_v[idx["gamma"][1]:idx["gamma"][2]], input_v[idx["t_s"][1]:idx["t_s"][2]], input_v[idx["c_0"][1]:idx["c_0"][2]], input_v[idx["T_0"][1]:idx["T_0"][2]])

        # Normalize engine inputs
        if settings.jet_mixing == true && settings.jet_shock == false
            V_j_star, rho_j_star, A_j_star, Tt_j_star = normalization_jet_mixing(settings, input_v[idx["V_j"][1]:idx["V_j"][2]], input_v[idx["rho_j"][1]:idx["rho_j"][2]], input_v[idx["A_j"][1]:idx["A_j"][2]], input_v[idx["Tt_j"][1]:idx["Tt_j"][2]], input_v[idx["c_0"][1]:idx["c_0"][2]], input_v[idx["rho_0"][1]:idx["rho_0"][2]], input_v[idx["T_0"][1]:idx["T_0"][2]])
            
        elseif settings.jet_shock == true && settings.jet_mixing == false
            V_j_star, A_j_star, Tt_j_star = normalization_jet_shock(settings, input_v[idx["V_j"][1]:idx["V_j"][2]], input_v[idx["A_j"][1]:idx["A_j"][2]], input_v[idx["Tt_j"][1]:idx["Tt_j"][2]], input_v[idx["c_0"][1]:idx["c_0"][2]], input_v[idx["T_0"][1]:idx["T_0"][2]])
            
        elseif settings.jet_shock ==true && settings.jet_mixing == true
            V_j_star, rho_j_star, A_j_star, Tt_j_star = normalization_jet(settings, input_v[idx["V_j"][1]:idx["V_j"][2]], input_v[idx["rho_j"][1]:idx["rho_j"][2]], input_v[idx["A_j"][1]:idx["A_j"][2]], input_v[idx["Tt_j"][1]:idx["Tt_j"][2]], input_v[idx["c_0"][1]:idx["c_0"][2]], input_v[idx["rho_0"][1]:idx["rho_0"][2]], input_v[idx["T_0"][1]:idx["T_0"][2]])
        end
        if settings.core
            if settings.method_core_turb == "GE"
                mdoti_c_star, Tti_c_star, Ttj_c_star, Pti_c_star, DTt_des_c_star = normalization_core_ge(settings, input_v[idx["mdoti_c"][1]:idx["mdoti_c"][2]], input_v[idx["Tti_c"][1]:idx["Tti_c"][2]], input_v[idx["Ttj_c"][1]:idx["Ttj_c"][2]], input_v[idx["Pti_c"][1]:idx["Pti_c"][2]], input_v[idx["DTt_des_c"][1]:idx["DTt_des_c"][2]], input_v[idx["c_0"][1]:idx["c_0"][2]], input_v[idx["rho_0"][1]:idx["rho_0"][2]], input_v[idx["T_0"][1]:idx["T_0"][2]], input_v[idx["p_0"][1]:idx["p_0"][2]])
                
            elseif settings.method_core_turb == "PW"
                mdoti_c_star, Tti_c_star, Ttj_c_star, Pti_c_star, rho_te_c_star, c_te_c_star, rho_ti_c_star, c_ti_c_star = normalization_core_pw(settings, input_v[idx["mdoti_c"][1]:idx["mdoti_c"][2]], input_v[idx["Tti_c"][1]:idx["Tti_c"][2]], input_v[idx["Ttj_c"][1]:idx["Ttj_c"][2]], input_v[idx["Pti_c"][1]:idx["Pti_c"][2]], input_v[idx["rho_te_c"][1]:idx["rho_te_c"][2]], input_v[idx["c_te_c"][1]:idx["c_te_c"][2]], input_v[idx["rho_ti_c"][1]:idx["rho_ti_c"][2]], input_v[idx["c_ti_c"][1]:idx["c_ti_c"][2]], input_v[idx["c_0"][1]:idx["c_0"][2]], input_v[idx["rho_0"][1]:idx["rho_0"][2]], input_v[idx["T_0"][1]:idx["T_0"][2]], input_v[idx["p_0"][1]:idx["p_0"][2]])
            end
        end
        if settings.fan_inlet==true || settings.fan_discharge==true
            DTt_f_star, mdot_f_star, N_f_star, A_f_star, d_f_star = normalization_fan(settings, input_v[idx["DTt_f"][1]:idx["DTt_f"][2]], input_v[idx["mdot_f"][1]:idx["mdot_f"][2]], input_v[idx["N_f"][1]:idx["N_f"][2]], input_v[idx["A_f"][1]:idx["A_f"][2]], input_v[idx["d_f"][1]:idx["d_f"][2]], input_v[idx["c_0"][1]:idx["c_0"][2]], input_v[idx["rho_0"][1]:idx["rho_0"][2]], input_v[idx["T_0"][1]:idx["T_0"][2]])
        end

        # Compute source
        if settings.fan_inlet
            spl .+= fan(settings, data, ac, n_t, shield, input_v[idx["M_0"][1]:idx["M_0"][2]], input_v[idx["c_0"][1]:idx["c_0"][2]], input_v[idx["T_0"][1]:idx["T_0"][2]], input_v[idx["rho_0"][1]:idx["rho_0"][2]], theta, DTt_f_star, mdot_f_star, N_f_star, A_f_star, d_f_star, "fan_inlet")
        end
        if settings.fan_discharge
            spl .+= fan(settings, data, ac, n_t, shield, input_v[idx["M_0"][1]:idx["M_0"][2]], input_v[idx["c_0"][1]:idx["c_0"][2]], input_v[idx["T_0"][1]:idx["T_0"][2]], input_v[idx["rho_0"][1]:idx["rho_0"][2]], theta, DTt_f_star, mdot_f_star, N_f_star, A_f_star, d_f_star, "fan_discharge")
        end

        if settings.core
            if settings.method_core_turb == "GE"
                msap_core = core_ge(settings, data, ac, n_t, input_v[idx["M_0"][1]:idx["M_0"][2]], theta, mdoti_c_star, Tti_c_star, Ttj_c_star, Pti_c_star, DTt_des_c_star)
            elseif settings.method.method_core_turb == "PW"
                msap_core = core_pw(settings, data, ac, n_t, input_v[idx["M_0"][1]:idx["M_0"][2]], theta, mdoti_c_star, Tti_c_star, Ttj_c_star, Pti_c_star, rho_te_c_star, c_te_c_star, rho_ti_c_star, c_ti_c_star)
            end
            
            if settings.suppression && settings.case_name in ["nasa_stca_standard", "stca_enginedesign_standard"]
                msap_core[idx_supp] = (10. ^(-2.3 / 10.) * msap_core)[idx_supp]
            end
            spl .+= msap_core
        end

        if settings.jet_mixing 
            msap_jet_mixing = jet_mixing(settings, data, ac, n_t, input_v[idx["M_0"][1]:idx["M_0"][2]], input_v[idx["c_0"][1]:idx["c_0"][2]], theta, V_j_star, rho_j_star, A_j_star, Tt_j_star)
            if settings.suppression && settings.case_name in ["nasa_stca_standard", "stca_enginedesign_standard"]
                msap_jet_mixing[idx_supp] = (10. ^(-2.3 / 10.) * msap_jet_mixing)[idx_supp]
            end
            spl .+= msap_jet_mixing
        end
        if settings.jet_shock
            msap_jet_shock = jet_shock(settings, data, ac, n_t, input_v[idx["M_0"][1]:idx["M_0"][2]], input_v[idx["c_0"][1]:idx["c_0"][2]], theta, V_j_star, input_v[idx["M_j"][1]:idx["M_j"][2]], A_j_star, Tt_j_star)
            if settings.suppression && settings.case_name in ["nasa_stca_standard", "stca_enginedesign_standard"]
                msap_jet_shock[idx_supp] = (10. ^(-2.3 / 10.) * msap_jet_shock)[idx_supp]
            end
            spl .+= msap_jet_shock
        end

        if settings.airframe
            spl .+= airframe(settings, data, ac, n_t, input_v[idx["M_0"][1]:idx["M_0"][2]], input_v[idx["mu_0"][1]:idx["mu_0"][2]], input_v[idx["c_0"][1]:idx["c_0"][2]], input_v[idx["rho_0"][1]:idx["rho_0"][2]], theta, phi, input_v[idx["theta_flaps"][1]:idx["theta_flaps"][2]], input_v[idx["I_landing_gear"][1]:idx["I_landing_gear"][2]])
        end

        # Compute propagation
        spl = propagation(settings, data, x_obs, spl, r, input_v[idx["x"][1]:idx["x"][2]], input_v[idx["z"][1]:idx["z"][2]], c_bar, input_v[idx["rho_0"][1]:idx["rho_0"][2]], input_v[idx["I_0"][1]:idx["I_0"][2]], beta)

        # Compute noise levels
        spl = f_spl(spl, input_v[idx["rho_0"][1]:idx["rho_0"][2]], input_v[idx["c_0"][1]:idx["c_0"][2]])
        
        oaspl = f_oaspl(spl)
        pnlt, C = f_pnlt(settings, data, spl)
        aspl = f_aspl(data, spl)

        # Compute integrated levels
        if settings.levels_int_metric == "ioaspl"
            level_int = f_ioaspl(t_o, oaspl)
        elseif settings.levels_int_metric == "ipnlt"
            level_int = f_ipnlt(t_o, pnlt)
        elseif settings.levels_int_metric == "epnl"
            level_int = f_epnl(t_o, pnlt)
        elseif settings.levels_int_metric == "sel"
            level_int = f_sel(t_o, aspl)
        end

        # Write output
        if objective == "noise"
            return level_int
        else
            return t_o, spl, aspl, oaspl, pnlt, C, level_int
        end

    end

    function noise_model(settings::PyObject, data::PyObject, ac::PyObject, n_t::Int64, idx::Dict{Any, Any}, objective::String, input_v::Union{Array{Float64, 1}, ReverseDiff.TrackedArray{Float64,Float64,1,Array{Float64,1},Array{Float64,1}}})

        # Get type of input vector
        T = eltype(input_v)

        # Number of observers
        n_obs = size(settings.x_observer_array)[1]

        # Compute noise for each observer
        t_o = zeros(T, (n_obs, n_t))
        spl = zeros(T, (n_obs, n_t, settings.N_f))
        aspl = zeros(T, (n_obs, n_t))
        oaspl = zeros(T, (n_obs, n_t))
        pnlt = zeros(T, (n_obs, n_t))
        C = zeros(T, (n_obs, n_t, settings.N_f))
        level_int = zeros(T, (n_obs, ))

        # Compute airframe shielding coefficients
        shield = shielding(settings, n_t)

        # Suppression  matrix
        idx_supp = findall(input_v[idx["TS"][1]:idx["TS"][2]].*ones(1, settings.N_f).>0.8)

        # Iterate over observers
        ncores=6
        # @floop ThreadedEx(basesize=n_obs÷ncores) 
        for i in range(1, n_obs, step=1)

            println("Computing noise at observer: ", i)

            if objective == "noise"
                level_int[i] = compute_noise_observer(settings, data, ac, n_t, idx, settings.x_observer_array[i,:], shield[i,:,:], idx_supp, objective, input_v)
            else
                t_o[i,:], spl[i,:,:], aspl[i,:], oaspl[i,:], pnlt[i,:], C[i,:,:], level_int[i] = compute_noise_observer(settings, data, ac, n_t, idx, settings.x_observer_array[i,:], shield[i,:,:], idx_supp, objective, input_v)
            end

        end

        # Write output
        if objective == "noise"
            k_smooth = 50.
            level_lateral = smooth_max(k_smooth, level_int[1:end-1])

            return [level_lateral, level_int[end]]
        else
            return t_o, spl, aspl, oaspl, pnlt, C, level_int
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
    X = vcat(X, 10. * ones(Float64, n_t))       # theta_flaps
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
            X = vcat(X, 30.   * ones(Float64, n_t))  # mdot_c
            X = vcat(X, 20.e6 * ones(Float64, n_t))  # Tti_c
            X = vcat(X, 800.  *  ones(Float64, n_t)) # Ttj_c
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
    Y = [100., 100.]
    J = Y.*X'
    #'

    # Define noise function
    noise_model_fwd = (x)->noise_model(settings, data, ac, n_t, idx, objective, x)

    return NoiseModel(settings, data, ac, n_t, idx, objective, noise_model, X, J, noise_model_fwd)
end

function setup(self::NoiseModel)
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
    push!(inputs, VarData("theta_flaps", shape=(n_t,), val=ones(n_t), units="deg"))

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
    if self.objective == "noise"
        push!(outputs, VarData("lateral", shape=(1, ), val=0.))
        push!(outputs, VarData("flyover", shape=(1, ), val=0.))
    else
        push!(outputs, VarData("t_o", shape=(n_obs, n_t), val=ones(n_obs, n_t)))
        push!(outputs, VarData("spl", shape=(n_obs, n_t, settings.N_f), val=ones(n_obs, n_t, settings.N_f)))
        push!(outputs, VarData("aspl", shape=(n_obs, n_t), val=ones(n_obs, n_t)))
        push!(outputs, VarData("oaspl", shape=(n_obs, n_t), val=ones(n_obs, n_t)))
        push!(outputs, VarData("pnlt", shape=(n_obs, n_t), val=ones(n_obs, n_t)))
        push!(outputs, VarData("C", shape=(n_obs, n_t, settings.N_f), val=ones(n_obs, n_t, settings.N_f)))
        push!(outputs, VarData(settings.levels_int_metric, shape=(n_obs, ), val=ones(n_obs, )))
    end

    ## Define partials --------------------------------------------------------------------------------
    partials = Vector{PartialsData}()
    
    if self.objective == "noise"
        for mic in ["lateral", "flyover"]
            push!(partials, PartialsData(mic, "x"))
            push!(partials, PartialsData(mic, "y"))
            push!(partials, PartialsData(mic, "z"))
            push!(partials, PartialsData(mic, "alpha"))
            push!(partials, PartialsData(mic, "gamma"))
            push!(partials, PartialsData(mic, "t_s"))
            push!(partials, PartialsData(mic, "rho_0"))
            push!(partials, PartialsData(mic, "mu_0"))
            push!(partials, PartialsData(mic, "c_0"))
            push!(partials, PartialsData(mic, "T_0"))
            push!(partials, PartialsData(mic, "p_0"))
            push!(partials, PartialsData(mic, "M_0"))
            push!(partials, PartialsData(mic, "I_0"))
            push!(partials, PartialsData(mic, "TS"))
            push!(partials, PartialsData(mic, "theta_flaps"))
            if settings.jet_mixing == true && settings.jet_shock == false
                push!(partials, PartialsData(mic, "V_j"))
                push!(partials, PartialsData(mic, "rho_j"))
                push!(partials, PartialsData(mic, "A_j"))
                push!(partials, PartialsData(mic, "Tt_j"))
            elseif settings.jet_shock == true && settings.jet_mixing == false
                push!(partials, PartialsData(mic, "V_j"))
                push!(partials, PartialsData(mic, "M_j"))
                push!(partials, PartialsData(mic, "A_j"))
                push!(partials, PartialsData(mic, "Tt_j"))
            elseif settings.jet_shock ==true && settings.jet_mixing == true
                push!(partials, PartialsData(mic, "V_j"))
                push!(partials, PartialsData(mic, "rho_j"))
                push!(partials, PartialsData(mic, "A_j"))
                push!(partials, PartialsData(mic, "Tt_j"))
                push!(partials, PartialsData(mic, "M_j"))
            end
            if settings.core
                if settings.method_core_turb == "GE"
                    push!(partials, PartialsData(mic, "mdoti_c"))
                    push!(partials, PartialsData(mic, "Tti_c"))
                    push!(partials, PartialsData(mic, "Ttj_c"))
                    push!(partials, PartialsData(mic, "Pti_c"))
                    push!(partials, PartialsData(mic, "DTt_des_c"))
                elseif settings.method_core_turb == "PW"
                    push!(partials, PartialsData(mic, "mdoti_c"))
                    push!(partials, PartialsData(mic, "Tti_c"))
                    push!(partials, PartialsData(mic, "Ttj_c"))
                    push!(partials, PartialsData(mic, "Pti_c"))
                    push!(partials, PartialsData(mic, "rho_te_c"))
                    push!(partials, PartialsData(mic, "c_te_c"))
                    push!(partials, PartialsData(mic, "rho_ti_c"))
                    push!(partials, PartialsData(mic, "c_ti_c"))
                end
            end
            if settings.airframe
                push!(partials, PartialsData(mic, "I_landing_gear"))
            end
            if settings.fan_inlet==true || settings.fan_discharge==true
                push!(partials, PartialsData(mic, "DTt_f"))
                push!(partials, PartialsData(mic, "mdot_f"))
                push!(partials, PartialsData(mic, "N_f"))
                push!(partials, PartialsData(mic, "A_f"))
                push!(partials, PartialsData(mic, "d_f"))
            end
        end
    end

    return inputs, outputs, partials
end

function get_noise_input_vector!(X::Array{Float64, 1}, settings::PyObject, inputs::PyDict{String, PyArray, true}, idx::Dict{Any, Any})

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
    X[idx["theta_flaps"][1]:idx["theta_flaps"][2]] = inputs["theta_flaps"]

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
        X[idx["I_landing_gear"][1]:idx["I_landing_gear"][2]] = inputs["I_landing_gear"]
    end
    if settings.fan_inlet==true || settings.fan_discharge==true
        X[idx["DTt_f"][1]:idx["DTt_f"][2]] = inputs["DTt_f"]
        X[idx["mdot_f"][1]:idx["mdot_f"][2]] = inputs["mdot_f"]
        X[idx["N_f"][1]:idx["N_f"][2]] = inputs["N_f"]
        X[idx["A_f"][1]:idx["A_f"][2]] = inputs["A_f"]
        X[idx["d_f"][1]:idx["d_f"][2]] = inputs["d_f"]
    end

end

function compute!(self::NoiseModel, inputs, outputs)
    # Load options
    settings = self.settings
    data = self.data
    ac = self.ac
    n_t = self.n_t
    idx = self.idx
    X = self.X

    # Get input vector
    get_noise_input_vector!(X, settings, inputs, idx)

    if self.objective == "noise"
        levels_int = self.noise_model(settings, data, ac, n_t, idx, self.objective, X)
        @. outputs["lateral"] = levels_int[1]
        @. outputs["flyover"] = levels_int[2]

        # Print inputs to file
        open(settings.pyNA_directory * "/cases/" * settings.case_name * "/output/" * settings.output_directory_name * "/" * "inputs_TS.txt","a") do io
            println(io, inputs["TS"])
        end

        open(settings.pyNA_directory * "/cases/" * settings.case_name * "/output/" * settings.output_directory_name * "/" * "inputs_alpha.txt","a") do io
            println(io, inputs["alpha"])
        end

        open(settings.pyNA_directory * "/cases/" * settings.case_name * "/output/" * settings.output_directory_name * "/" * "inputs_theta_flaps.txt","a") do io
            println(io, inputs["theta_flaps"])
        end
        
        # Print outputs to file
        open(settings.pyNA_directory * "/cases/" * settings.case_name * "/output/" * settings.output_directory_name * "/" * "outputs_levels_int.txt", "a") do io
            println(io, levels_int)
        end

    else
        t_o, spl, aspl, oaspl, pnlt, C, levels_int = self.noise_model(settings, data, ac, n_t, idx, self.objective, X)
        @. outputs["t_o"] = t_o
        @. outputs["spl"] = spl
        @. outputs["aspl"] = aspl
        @. outputs["oaspl"] = oaspl
        @. outputs["pnlt"] = pnlt
        @. outputs["C"] = C
        @. outputs[settings.levels_int_metric] = levels_int
    end

end

function compute_partials!(self::NoiseModel, inputs, partials)
    # Load options
    settings = self.settings
    n_t = self.n_t
    idx = self.idx
    X = self.X
    J = self.J

    # Print start statement
    println("Computing partials noise")

    # Get input vector
    get_noise_input_vector!(X, settings, inputs, idx)

    # compute Jacobian
    ReverseDiff.jacobian!(J, self.noise_model_fwd, X)
    
    for (i, mic) in enumerate(["lateral", "flyover"])
        dnoise_dx = reshape(J[i, idx["x"][1]:idx["x"][2]], (1, n_t))
        @. partials[mic, "x"] = dnoise_dx
        dnoise_dy = reshape(J[i, idx["y"][1]:idx["y"][2]], (1, n_t))
        @. partials[mic, "y"] = dnoise_dy
        dnoise_dz = reshape(J[i, idx["z"][1]:idx["z"][2]], (1, n_t))
        @. partials[mic, "z"] = dnoise_dz
        dnoise_dalpha = reshape(J[i, idx["alpha"][1]:idx["alpha"][2]], (1, n_t))
        @. partials[mic, "alpha"] = dnoise_dalpha
        dnoise_dgamma = reshape(J[i, idx["gamma"][1]:idx["gamma"][2]], (1, n_t))
        @. partials[mic, "gamma"] = dnoise_dgamma
        dnoise_dts = reshape(J[i, idx["t_s"][1]:idx["t_s"][2]], (1, n_t))
        @. partials[mic, "t_s"] = dnoise_dts
        dnoise_drho_0 = reshape(J[i, idx["rho_0"][1]:idx["rho_0"][2]], (1, n_t))
        @. partials[mic, "rho_0"] = dnoise_drho_0
        dnoise_dmu_0 = reshape(J[i, idx["mu_0"][1]:idx["mu_0"][2]], (1, n_t))
        @. partials[mic, "mu_0"] = dnoise_dmu_0
        dnoise_dc_0 = reshape(J[i, idx["c_0"][1]:idx["c_0"][2]], (1, n_t))
        @. partials[mic, "c_0"] = dnoise_dc_0
        dnoise_dT_0 = reshape(J[i, idx["T_0"][1]:idx["T_0"][2]], (1, n_t))
        @. partials[mic, "T_0"] = dnoise_dT_0
        dnoise_dp_0 = reshape(J[i, idx["p_0"][1]:idx["p_0"][2]], (1, n_t))
        @. partials[mic, "p_0"] = dnoise_dp_0
        dnoise_dM_0 = reshape(J[i, idx["M_0"][1]:idx["M_0"][2]], (1, n_t))
        @. partials[mic, "M_0"] = dnoise_dM_0
        dnoise_dI_0 = reshape(J[i, idx["I_0"][1]:idx["I_0"][2]], (1, n_t))
        @. partials[mic, "I_0"] = dnoise_dI_0
        dnoise_dTS = reshape(J[i, idx["TS"][1]:idx["TS"][2]], (1, n_t))
        @. partials[mic, "TS"] = dnoise_dTS
        dnoise_dtheta_flaps = reshape(J[i, idx["theta_flaps"][1]:idx["theta_flaps"][2]], (1, n_t))
        @. partials[mic, "theta_flaps"] = dnoise_dtheta_flaps
        if settings.jet_mixing == true && settings.jet_shock == false
            dnoise_dvj = reshape(J[i, idx["V_j"][1]:idx["V_j"][2]], (1, n_t))
            @. partials[mic, "V_j"] = dnoise_dvj
            dnoise_drhoj = reshape(J[i, idx["rho_j"][1]:idx["rho_j"][2]], (1, n_t))
            @. partials[mic, "rho_j"] = dnoise_drhoj
            dnoise_dAj = reshape(J[i, idx["A_j"][1]:idx["A_j"][2]], (1, n_t))
            @. partials[mic, "A_j"] = dnoise_dAj
            dnoise_dTtj = reshape(J[i, idx["Tt_j"][1]:idx["Tt_j"][2]], (1, n_t))
            @. partials[mic, "Tt_j"] = dnoise_dTtj
        elseif settings.jet_shock == true && settings.jet_mixing == false
            dnoise_dvj = reshape(J[i, idx["V_j"][1]:idx["V_j"][2]], (1, n_t))
            @. partials[mic, "V_j"] = dnoise_dvj
            dnoise_dMj = reshape(J[i, idx["M_j"][1]:idx["M_j"][2]], (1, n_t))
            @. partials[mic, "M_j"] = dnoise_dMj
            dnoise_dAj = reshape(J[i, idx["A_j"][1]:idx["A_j"][2]], (1, n_t))
            @. partials[mic, "A_j"] = dnoise_dAj
            dnoise_dTtj = reshape(J[i, idx["Tt_j"][1]:idx["Tt_j"][2]], (1, n_t))
            @. partials[mic, "Tt_j"] = dnoise_dTtj
        elseif settings.jet_shock ==true && settings.jet_mixing == true
            dnoise_dvj = reshape(J[i, idx["V_j"][1]:idx["V_j"][2]], (1, n_t))
            @. partials[mic, "V_j"] = dnoise_dvj
            dnoise_drhoj = reshape(J[i, idx["rho_j"][1]:idx["rho_j"][2]], (1, n_t))
            @. partials[mic, "rho_j"] = dnoise_drhoj
            dnoise_dAj = reshape(J[i, idx["A_j"][1]:idx["A_j"][2]], (1, n_t))
            @. partials[mic, "A_j"] = dnoise_dAj
            dnoise_dTtj = reshape(J[i, idx["Tt_j"][1]:idx["Tt_j"][2]], (1, n_t))
            @. partials[mic, "Tt_j"] = dnoise_dTtj
            dnoise_dMj = reshape(J[i, idx["M_j"][1]:idx["M_j"][2]], (1, n_t))
            @. partials[mic, "M_j"] = dnoise_dMj
        end
        if settings.core
            if settings.method_core_turb == "GE"
                dnoise_dmdoti_c = reshape(J[i, idx["mdoti_c"][1]:idx["mdoti_c"][2]], (1, n_t))
                @. partials[mic, "mdoti_c"] = dnoise_dmdoti_c
                dnoise_dTti_c = reshape(J[i, idx["Tti_c"][1]:idx["Tti_c"][2]], (1, n_t))
                @. partials[mic, "Tti_c"] = dnoise_dTti_c
                dnoise_dTtj_c = reshape(J[i, idx["Ttj_c"][1]:idx["Ttj_c"][2]], (1, n_t))
                @. partials[mic, "Ttj_c"] = dnoise_dTtj_c
                dnoise_dPti_c = reshape(J[i, idx["Pti_c"][1]:idx["Pti_c"][2]], (1, n_t))
                @. partials[mic, "Pti_c"] = dnoise_dPti_c
                dnoise_dDTt_des_c = reshape(J[i, idx["DTt_des_c"][1]:idx["DTt_des_c"][2]], (1, n_t))
                @. partials[mic, "DTt_des_c"] = dnoise_dDTt_des_c
            elseif settings.method_core_turb == "PW"
                dnoise_dmdoti_c = reshape(J[i, idx["mdoti_c"][1]:idx["mdoti_c"][2]], (1, n_t))
                @. partials[mic, "mdoti_c"] = dnoise_dmdoti_c
                dnoise_dTti_c = reshape(J[i, idx["Tti_c"][1]:idx["Tti_c"][2]], (1, n_t))
                @. partials[mic, "Tti_c"] = dnoise_dTti_c
                dnoise_dTtj_c = reshape(J[i, idx["Ttj_c"][1]:idx["Ttj_c"][2]], (1, n_t))
                @. partials[mic, "Ttj_c"] = dnoise_dTtj_c
                dnoise_dPti_c = reshape(J[i, idx["Pti_c"][1]:idx["Pti_c"][2]], (1, n_t))
                @. partials[mic, "Pti_c"] = dnoise_dPti_c
                dnoise_drho_te_c = reshape(J[i, idx["rho_te_c"][1]:idx["rho_te_c"][2]], (1, n_t))
                @. partials[mic, "rho_te_c"] = dnoise_drho_te_c
                dnoise_dc_te_c = reshape(J[i, idx["c_te_c"][1]:idx["c_te_c"][2]], (1, n_t))
                @. partials[mic, "c_te_c"] = dnoise_dc_te_c
                dnoise_drho_ti_c = reshape(J[i, idx["rho_ti_c"][1]:idx["rho_ti_c"][2]], (1, n_t))
                @. partials[mic, "rho_ti_c"] = dnoise_drho_ti_c
                dnoise_dc_ti_c = reshape(J[i, idx["c_ti_c"][1]:idx["c_ti_c"][2]], (1, n_t))
                @. partials[mic, "c_ti_c"] = dnoise_dc_ti_c
            end
        end
        if settings.airframe
            dnoise_dI_landing_gear = reshape(J[i, idx["I_landing_gear"][1]:idx["I_landing_gear"][2]], (1, n_t))
            @. partials[mic, "I_landing_gear"] = dnoise_dI_landing_gear
        end
        if settings.fan_inlet==true || settings.fan_discharge==true
            dnoise_dDTt_f = reshape(J[i, idx["DTt_f"][1]:idx["DTt_f"][2]], (1, n_t))
            @. partials[mic, "DTt_f"] = dnoise_dDTt_f
            dnoise_dmdot_f = reshape(J[i, idx["mdot_f"][1]:idx["mdot_f"][2]], (1, n_t))
            @. partials[mic, "mdot_f"] = dnoise_dmdot_f
            dnoise_dN_f = reshape(J[i, idx["N_f"][1]:idx["N_f"][2]], (1, n_t))
            @. partials[mic, "N_f"] = dnoise_dN_f
            dnoise_dA_f = reshape(J[i, idx["A_f"][1]:idx["A_f"][2]], (1, n_t))
            @. partials[mic, "A_f"] = dnoise_dA_f
            dnoise_dd_f = reshape(J[i, idx["d_f"][1]:idx["d_f"][2]], (1, n_t))
            @. partials[mic, "d_f"] = dnoise_dd_f
        end
    end

    println("Done computing partials noise.")

end