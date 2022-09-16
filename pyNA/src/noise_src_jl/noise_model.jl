using OpenMDAO: AbstractExplicitComp, VarData, PartialsData, get_rows_cols
import OpenMDAO: setup, compute!, compute_partials!
using Interpolations
using ReverseDiff: JacobianTape, jacobian!, compile
using LinearAlgebra
using ConcreteStructs
using PCHIPInterpolation
using Dates
using BenchmarkTools
include("get_noise.jl")


# Define propagation struct
@concrete struct NoiseModel <: AbstractExplicitComp
    settings
    pyna_ip
    data
    sealevel_atmosphere
    airframe 
    n_t :: Int64
    idx
    objective :: String
    X :: Array{Float64, 1}
    J :: Array{Float64, 2}
    get_nosie_fwd!
end

struct PynaInterpolations
    f_supp_fi  # Fan
    f_supp_fd
    f_F3IB
    f_F3DB
    f_F3TI
    f_F3TD
    f_F2CT
    f_TCS_takeoff_ih1
    f_TCS_takeoff_ih2
    f_TCS_approach_ih1
    f_TCS_approach_ih2
    f_D_core
    f_S_core
    f_omega_jet
    f_log10P_jet
    f_log10D_jet
    f_xi_jet
    f_log10F_jet
    f_m_theta_jet
    f_C_jet  # Jet shock
    f_H_jet
    f_hsr_supp  # Airframe suppression
    f_abs # Propagation
    f_faddeeva_real
    f_faddeeva_imag
    f_noy  # Noy
    f_aw
end

function NoiseModel(settings, data, sealevel_atmosphere, airframe, n_t, idx, objective)

    # Default values for input vector
    X = range(1, 10000, length=n_t)             # x
    X = vcat(X, zeros(Float64, n_t))            # y
    X = vcat(X, range(1, 1000, length=n_t))     # z
    X = vcat(X, 10. * ones(Float64, n_t))       # alpha
    X = vcat(X, 10. * ones(Float64, n_t))       # gamma
    X = vcat(X, range(0, 100, length=n_t))      # t_s
    X = vcat(X, 0.3 * ones(Float64, n_t))       # M_0
    if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
        X = vcat(X, 1. * ones(Float64, n_t))        # TS
    end
    if settings["atmosphere_type"] == "stratified"
        X = vcat(X, 340.294 * ones(Float64, n_t))   # c_0
        X = vcat(X, 288.15 * ones(Float64, n_t))    # T_0
        X = vcat(X, 1.225 * ones(Float64, n_t))     # rho_0
        X = vcat(X, 101325. * ones(Float64, n_t))   # p_0
        X = vcat(X, 1.789e-5 * ones(Float64, n_t))  # mu_0
        X = vcat(X, 400. * ones(Float64, n_t))      # I_0
    end
    if settings["fan_inlet_source"]==true || settings["fan_discharge_source"]==true
        X = vcat(X, 70.   * ones(Float64, n_t))  # DTt_f
        X = vcat(X, 200.  * ones(Float64, n_t))  # mdot_f
        X = vcat(X, 8000. * ones(Float64, n_t))  # N_f
        X = vcat(X, 0.9   * ones(Float64, n_t))  # A_f
        X = vcat(X, 1.7   * ones(Float64, n_t))  # d_f
    end
    if settings["core_source"]
        if settings["core_turbine_attenuation_method"] == "ge"
            X = vcat(X, 30.   * ones(Float64, n_t))  # mdot_c
            X = vcat(X, 20.e6 * ones(Float64, n_t))  # Tti_c
            X = vcat(X, 800.  *  ones(Float64, n_t)) # Ttj_c
            X = vcat(X, 1600. * ones(Float64, n_t))  # Pti_c
            X = vcat(X, 800.  * ones(Float64, n_t))  # DTt_des_c
        elseif settings["core_turbine_attenuation_method"] == "pw"
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
    if settings["jet_mixing_source"] == true && settings["jet_shock_source"] == false
        X = vcat(X, 400. * ones(Float64, n_t))  # V_j
        X = vcat(X, 0.8  * ones(Float64, n_t))  # rho_j
        X = vcat(X, 0.5  * ones(Float64, n_t))  # A_j
        X = vcat(X, 500. * ones(Float64, n_t))  # Tt_j
    elseif settings["jet_shock_source"] == true && settings["jet_mixing_source"] == false
        X = vcat(X, 400. * ones(Float64, n_t))  # V_j
        X = vcat(X, 0.5  * ones(Float64, n_t))  # A_j
        X = vcat(X, 500. * ones(Float64, n_t))  # Tt_j
        X = vcat(X, 1.   * ones(Float64, n_t))  # M_j
    elseif settings["jet_shock_source"] ==true && settings["jet_mixing_source"] == true
        X = vcat(X, 400. * ones(Float64, n_t))  # V_j
        X = vcat(X, 0.8  * ones(Float64, n_t))  # rho_j
        X = vcat(X, 0.5  * ones(Float64, n_t))  # A_j
        X = vcat(X, 500. * ones(Float64, n_t))  # Tt_j
        X = vcat(X, 1.   * ones(Float64, n_t))  # M_j
    end
    if settings["airframe_source"]
        X = vcat(X, 10. * ones(Float64, n_t))  # theta_flaps
        X = vcat(X, ones(Float64, n_t))        # I_landing_gear
    end
    
    # Default values for output value and jacobian
    Y = [100., 100.]
    J = Y.*X'
    #'

    # Get interpolation functions
    f_supp_fi, f_supp_fd, f_F3IB, f_F3DB, f_F3TI, f_F3TD, f_F2CT, f_TCS_takeoff_ih1, f_TCS_takeoff_ih2, f_TCS_approach_ih1, f_TCS_approach_ih2 = get_fan_interpolation_functions(settings, data)
    f_D_core, f_S_core = get_core_interpolation_functions()
    f_omega_jet, f_log10P_jet, f_log10D_jet, f_xi_jet, f_log10F_jet, f_m_theta_jet = get_jet_mixing_interpolation_functions(data)
    f_C_jet, f_H_jet = get_jet_shock_interpolation_functions()
    f_hsr_supp = get_airframe_interpolation_functions(data)
    f_abs = get_atmospheric_absorption_interpolation_functions(data)
    f_faddeeva_real, f_faddeeva_imag = get_ground_effects_interpolation_functions(data)
    f_noy = get_noy_interpolation_functions(data)
    f_aw = get_a_weighting_interpolation_functions(data)
    pyna_ip = PynaInterpolations(f_supp_fi, f_supp_fd, f_F3IB, f_F3DB, f_F3TI, f_F3TD, f_F2CT, f_TCS_takeoff_ih1, f_TCS_takeoff_ih2, f_TCS_approach_ih1, f_TCS_approach_ih2, f_D_core, f_S_core, f_omega_jet, f_log10P_jet, f_log10D_jet, f_xi_jet, f_log10F_jet, f_m_theta_jet, f_C_jet, f_H_jet, f_hsr_supp, f_abs, f_faddeeva_real, f_faddeeva_imag, f_noy, f_aw)

    # Jacobian configuration
    # if objective == "noise"
    get_noise_fwd! = (y,x)->get_noise!(y, x, settings, pyna_ip, airframe, data, sealevel_atmosphere, idx)
    # model_tape = JacobianTape(get_noise_fwd!, Y, X)
    # compiled_model_tape = compile(model_tape)
    # else
    # compiled_model_tape = nothing
    # end

    return NoiseModel(settings, pyna_ip, data, sealevel_atmosphere, airframe, n_t, idx, objective, X, J, get_nosie_fwd!)
end

function setup(self::NoiseModel)
    # Load options
    settings = self.settings
    n_t = self.n_t

    # Number of observers
    n_obs = size(settings["x_observer_array"])[1]

    # Define inputs --------------------------------------------------------------------------------
    inputs = Vector{VarData}()
    push!(inputs, VarData("x", shape=(n_t, ), val=ones(n_t), units="m"))
    push!(inputs, VarData("y", shape=(n_t, ), val=ones(n_t), units="m"))
    push!(inputs, VarData("z", shape=(n_t, ), val=ones(n_t), units="m"))
    push!(inputs, VarData("alpha", shape=(n_t, ), val=ones(n_t), units="deg"))
    push!(inputs, VarData("gamma", shape=(n_t, ), val=ones(n_t), units="deg"))
    push!(inputs, VarData("t_s", shape=(n_t, ), val=ones(n_t), units="s"))
    push!(inputs, VarData("M_0", shape=(n_t,), val=ones(n_t)))
    if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
        push!(inputs, VarData("TS", shape=(n_t,), val=ones(n_t)))
    end
    if settings["atmosphere_type"] == "stratified"
        push!(inputs, VarData("c_0", shape=(n_t, ), val=ones(n_t), units="m/s"))
        push!(inputs, VarData("T_0", shape=(n_t, ), val=ones(n_t), units="K"))
        push!(inputs, VarData("rho_0", shape=(n_t,), val=ones(n_t), units="kg/m**3"))
        push!(inputs, VarData("p_0", shape=(n_t, ), val=ones(n_t), units="Pa"))
        push!(inputs, VarData("mu_0", shape=(n_t,), val=ones(n_t), units="kg/m/s"))
        push!(inputs, VarData("I_0", shape=(n_t,), val=ones(n_t), units="kg/m**2/s"))    
    end
    if settings["fan_inlet_source"]==true || settings["fan_discharge_source"]==true
        push!(inputs, VarData("DTt_f", shape=(n_t,), val=ones(n_t), units="K"))
        push!(inputs, VarData("mdot_f", shape=(n_t,), val=ones(n_t), units="kg/s"))
        push!(inputs, VarData("N_f", shape=(n_t,), val=ones(n_t), units="rpm"))
        push!(inputs, VarData("A_f", shape=(n_t,), val=ones(n_t), units="m**2"))
        push!(inputs, VarData("d_f", shape=(n_t,), val=ones(n_t), units="m"))
    end
    if settings["core_source"]
        if settings["core_turbine_attenuation_method"] == "ge"
            push!(inputs, VarData("mdoti_c", shape=(n_t,), val=ones(n_t), units="kg/s"))
            push!(inputs, VarData("Tti_c", shape=(n_t,), val=ones(n_t), units="K"))
            push!(inputs, VarData("Ttj_c", shape=(n_t,), val=ones(n_t), units="K"))
            push!(inputs, VarData("Pti_c", shape=(n_t,), val=ones(n_t), units="Pa"))
            push!(inputs, VarData("DTt_des_c", shape=(n_t,), val=ones(n_t), units="K"))
        elseif settings["core_turbine_attenuation_method"] == "pw"
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
    if settings["jet_mixing_source"] == true && settings["jet_shock_source"] == false
        push!(inputs, VarData("V_j", shape=(n_t,), val=ones(n_t), units="m/s"))
        push!(inputs, VarData("rho_j", shape=(n_t,), val=ones(n_t), units="kg/m**3"))
        push!(inputs, VarData("A_j", shape=(n_t,), val=ones(n_t), units="m**2"))
        push!(inputs, VarData("Tt_j", shape=(n_t,), val=ones(n_t), units="K"))
    elseif settings["jet_shock_source"] == true && settings["jet_mixing_source"] == false
        push!(inputs, VarData("V_j", shape=(n_t,), val=ones(n_t), units="m/s"))
        push!(inputs, VarData("M_j", shape=(n_t,), val=ones(n_t)))
        push!(inputs, VarData("A_j", shape=(n_t,), val=ones(n_t), units="m**2"))
        push!(inputs, VarData("Tt_j", shape=(n_t,), val=ones(n_t), units="K"))
    elseif settings["jet_shock_source"] ==true && settings["jet_mixing_source"] == true
        push!(inputs, VarData("V_j", shape=(n_t,), val=ones(n_t), units="m/s"))
        push!(inputs, VarData("rho_j", shape=(n_t,), val=ones(n_t), units="kg/m**3"))
        push!(inputs, VarData("A_j", shape=(n_t,), val=ones(n_t), units="m**2"))
        push!(inputs, VarData("Tt_j", shape=(n_t,), val=ones(n_t), units="K"))
        push!(inputs, VarData("M_j", shape=(n_t,), val=ones(n_t)))
    end
    if settings["airframe_source"]
        push!(inputs, VarData("theta_flaps", shape=(n_t,), val=ones(n_t), units="deg"))
        push!(inputs, VarData("I_landing_gear", shape=(n_t,), val=ones(n_t)))
    end

    # Define outputs --------------------------------------------------------------------------------
    outputs = Vector{VarData}()
    # if self.objective == "noise"
    push!(outputs, VarData("lateral", shape=(1, ), val=0.))
    push!(outputs, VarData("flyover", shape=(1, ), val=0.))
    # else
        # push!(outputs, VarData("t_o", shape=(n_obs, n_t), val=ones(n_obs, n_t)))
        # push!(outputs, VarData("spl", shape=(n_obs, n_t, settings["n_frequency_bands"]), val=ones(n_obs, n_t, settings["n_frequency_bands"])))
        # push!(outputs, VarData("level", shape=(n_obs, n_t), val=ones(n_obs, n_t)))
        # push!(outputs, VarData(settings["levels_int_metric"], shape=(n_obs, ), val=ones(n_obs, )))
    # end

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
            push!(partials, PartialsData(mic, "M_0"))
            if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
                push!(partials, PartialsData(mic, "TS"))
            end
            if settings["atmosphere_type"] == "stratified"
                push!(partials, PartialsData(mic, "c_0"))
                push!(partials, PartialsData(mic, "T_0"))
                push!(partials, PartialsData(mic, "rho_0"))
                push!(partials, PartialsData(mic, "p_0"))
                push!(partials, PartialsData(mic, "mu_0"))
                push!(partials, PartialsData(mic, "I_0"))
            end
            if settings["fan_inlet_source"]==true || settings["fan_discharge_source"]==true
                push!(partials, PartialsData(mic, "DTt_f"))
                push!(partials, PartialsData(mic, "mdot_f"))
                push!(partials, PartialsData(mic, "N_f"))
                push!(partials, PartialsData(mic, "A_f"))
                push!(partials, PartialsData(mic, "d_f"))
            end
            if settings["core_source"]
                if settings["core_turbine_attenuation_method"] == "ge"
                    push!(partials, PartialsData(mic, "mdoti_c"))
                    push!(partials, PartialsData(mic, "Tti_c"))
                    push!(partials, PartialsData(mic, "Ttj_c"))
                    push!(partials, PartialsData(mic, "Pti_c"))
                    push!(partials, PartialsData(mic, "DTt_des_c"))
                elseif settings["core_turbine_attenuation_method"] == "pw"
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
            if settings["jet_mixing_source"] == true && settings["jet_shock_source"] == false
                push!(partials, PartialsData(mic, "V_j"))
                push!(partials, PartialsData(mic, "rho_j"))
                push!(partials, PartialsData(mic, "A_j"))
                push!(partials, PartialsData(mic, "Tt_j"))
            elseif settings["jet_shock_source"] == true && settings["jet_mixing_source"] == false
                push!(partials, PartialsData(mic, "V_j"))
                push!(partials, PartialsData(mic, "M_j"))
                push!(partials, PartialsData(mic, "A_j"))
                push!(partials, PartialsData(mic, "Tt_j"))
            elseif settings["jet_shock_source"] ==true && settings["jet_mixing_source"] == true
                push!(partials, PartialsData(mic, "V_j"))
                push!(partials, PartialsData(mic, "rho_j"))
                push!(partials, PartialsData(mic, "A_j"))
                push!(partials, PartialsData(mic, "Tt_j"))
                push!(partials, PartialsData(mic, "M_j"))
            end 
            if settings["airframe_source"]
                push!(partials, PartialsData(mic, "theta_flaps"))
                push!(partials, PartialsData(mic, "I_landing_gear"))
            end
            
        end
    end

    return inputs, outputs, partials
end

function get_noise_input_vector!(X::Array{Float64, 1}, settings, inputs::PyDict{String, PyArray, true}, idx::Dict{Any, Any})

    # Extract inputs
    X[idx["x"][1]:idx["x"][2]] = inputs["x"]
    X[idx["y"][1]:idx["y"][2]] = inputs["y"]
    X[idx["z"][1]:idx["z"][2]] = inputs["z"]
    X[idx["alpha"][1]:idx["alpha"][2]] = inputs["alpha"]
    X[idx["gamma"][1]:idx["gamma"][2]] = inputs["gamma"]
    X[idx["t_s"][1]:idx["t_s"][2]] = inputs["t_s"]
    X[idx["M_0"][1]:idx["M_0"][2]] = inputs["M_0"]
    if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]
        X[idx["TS"][1]:idx["TS"][2]] = inputs["TS"]
    end
    if settings["atmosphere_type"] == "stratified"
        X[idx["c_0"][1]:idx["c_0"][2]] = inputs["c_0"]
        X[idx["T_0"][1]:idx["T_0"][2]] = inputs["T_0"]
        X[idx["rho_0"][1]:idx["rho_0"][2]] = inputs["rho_0"]
        X[idx["p_0"][1]:idx["p_0"][2]] = inputs["p_0"]
        X[idx["mu_0"][1]:idx["mu_0"][2]] = inputs["mu_0"]
        X[idx["I_0"][1]:idx["I_0"][2]] = inputs["I_0"]
    end
    if settings["fan_inlet_source"]==true || settings["fan_discharge_source"]==true
        X[idx["DTt_f"][1]:idx["DTt_f"][2]] = inputs["DTt_f"]
        X[idx["mdot_f"][1]:idx["mdot_f"][2]] = inputs["mdot_f"]
        X[idx["N_f"][1]:idx["N_f"][2]] = inputs["N_f"]
        X[idx["A_f"][1]:idx["A_f"][2]] = inputs["A_f"]
        X[idx["d_f"][1]:idx["d_f"][2]] = inputs["d_f"]
    end
    if settings["core_source"]
        if settings["core_turbine_attenuation_method"] == "ge"
            X[idx["mdoti_c"][1]:idx["mdoti_c"][2]] = inputs["mdoti_c"]
            X[idx["Tti_c"][1]:idx["Tti_c"][2]] = inputs["Tti_c"]
            X[idx["Ttj_c"][1]:idx["Ttj_c"][2]] = inputs["Ttj_c"]
            X[idx["Pti_c"][1]:idx["Pti_c"][2]] = inputs["Pti_c"]
            X[idx["DTt_des_c"][1]:idx["DTt_des_c"][2]] = inputs["DTt_des_c"]
        elseif settings["core_turbine_attenuation_method"] == "pw"
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
    if settings["jet_mixing_source"] == true && settings["jet_shock_source"] == false
        X[idx["V_j"][1]:idx["V_j"][2]] = inputs["V_j"]
        X[idx["rho_j"][1]:idx["rho_j"][2]] = inputs["rho_j"]
        X[idx["A_j"][1]:idx["A_j"][2]] = inputs["A_j"]
        X[idx["Tt_j"][1]:idx["Tt_j"][2]] = inputs["Tt_j"]
    elseif settings["jet_shock_source"] == true && settings["jet_mixing_source"] == false
        X[idx["V_j"][1]:idx["V_j"][2]] = inputs["V_j"]
        X[idx["M_j"][1]:idx["M_j"][2]] = inputs["M_j"]
        X[idx["A_j"][1]:idx["A_j"][2]] = inputs["A_j"]
        X[idx["Tt_j"][1]:idx["Tt_j"][2]] = inputs["Tt_j"]
    elseif settings["jet_shock_source"] ==true && settings["jet_mixing_source"] == true
        X[idx["V_j"][1]:idx["V_j"][2]] = inputs["V_j"]
        X[idx["rho_j"][1]:idx["rho_j"][2]] = inputs["rho_j"]
        X[idx["A_j"][1]:idx["A_j"][2]] = inputs["A_j"]
        X[idx["Tt_j"][1]:idx["Tt_j"][2]] = inputs["Tt_j"]
        X[idx["M_j"][1]:idx["M_j"][2]] = inputs["M_j"]
    end
    if settings["airframe_source"]
        X[idx["theta_flaps"][1]:idx["theta_flaps"][2]] = inputs["theta_flaps"]
        X[idx["I_landing_gear"][1]:idx["I_landing_gear"][2]] = inputs["I_landing_gear"]
    end

end

function compute!(self::NoiseModel, inputs, outputs)

    # Get input vector
    get_noise_input_vector!(self.X, self.settings, inputs, self.idx)

    # if self.objective == "noise"
    levels_int = zeros(2)

    get_noise!(levels_int, self.X, self.settings, self.pyna_ip, self.airframe, self.data, self.sealevel_atmosphere, self.idx)
    @. outputs["lateral"] = levels_int[1]
    @. outputs["flyover"] = levels_int[2]
    
        # Print outputs to file
        # open(self.settings["pyna_directory"] * "/cases/" * self.settings["case_name"] * "/output/" * self.settings["output_directory_name"] * "/" * "outputs_levels_int.txt", "a") do io
            # println(io, levels_int)
        # end

    # else
        # t_o, spl, level, levels_int = get_noise(self.X, self.settings, self.pyna_ip, self.airframe, self.data, self.sealevel_atmosphere, self.idx)
        # @. outputs["t_o"] = t_o
        # @. outputs["spl"] = spl
        # @. outputs["level"] = level
        # @. outputs[self.settings["levels_int_metric"]] = levels_int
    # end

    # Try computing gradients
    println("--- Computing gradients ---")
    jacobian!(self.J, self.get_nosie_fwd!, self.X)
    println(J)

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
    jacobian!(J, self.noise_model_fwd, X)
    
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
        dnoise_dM_0 = reshape(J[i, idx["M_0"][1]:idx["M_0"][2]], (1, n_t))
        @. partials[mic, "M_0"] = dnoise_dM_0
        if settings["core_jet_suppression"] && settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"]        
            dnoise_dTS = reshape(J[i, idx["TS"][1]:idx["TS"][2]], (1, n_t))
            @. partials[mic, "TS"] = dnoise_dTS
        end
        if settings["atmosphere_type"] == "stratified"
            dnoise_dc_0 = reshape(J[i, idx["c_0"][1]:idx["c_0"][2]], (1, n_t))
            @. partials[mic, "c_0"] = dnoise_dc_0
            dnoise_dT_0 = reshape(J[i, idx["T_0"][1]:idx["T_0"][2]], (1, n_t))
            @. partials[mic, "T_0"] = dnoise_dT_0
            dnoise_drho_0 = reshape(J[i, idx["rho_0"][1]:idx["rho_0"][2]], (1, n_t))
            @. partials[mic, "rho_0"] = dnoise_drho_0
            dnoise_dp_0 = reshape(J[i, idx["p_0"][1]:idx["p_0"][2]], (1, n_t))
            @. partials[mic, "p_0"] = dnoise_dp_0
            dnoise_dmu_0 = reshape(J[i, idx["mu_0"][1]:idx["mu_0"][2]], (1, n_t))
            @. partials[mic, "mu_0"] = dnoise_dmu_0
            dnoise_dI_0 = reshape(J[i, idx["I_0"][1]:idx["I_0"][2]], (1, n_t))
            @. partials[mic, "I_0"] = dnoise_dI_0
        end
        if settings["fan_inlet_source"]==true || settings["fan_discharge_source"]==true
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
        if settings["core_source"]
            if settings["core_turbine_attenuation_method"] == "ge"
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
            elseif settings["core_turbine_attenuation_method"] == "pw"
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
        if settings["jet_mixing_source"] == true && settings["jet_shock_source"] == false
            dnoise_dvj = reshape(J[i, idx["V_j"][1]:idx["V_j"][2]], (1, n_t))
            @. partials[mic, "V_j"] = dnoise_dvj
            dnoise_drhoj = reshape(J[i, idx["rho_j"][1]:idx["rho_j"][2]], (1, n_t))
            @. partials[mic, "rho_j"] = dnoise_drhoj
            dnoise_dAj = reshape(J[i, idx["A_j"][1]:idx["A_j"][2]], (1, n_t))
            @. partials[mic, "A_j"] = dnoise_dAj
            dnoise_dTtj = reshape(J[i, idx["Tt_j"][1]:idx["Tt_j"][2]], (1, n_t))
            @. partials[mic, "Tt_j"] = dnoise_dTtj
        elseif settings["jet_shock_source"] == true && settings["jet_mixing_source"] == false
            dnoise_dvj = reshape(J[i, idx["V_j"][1]:idx["V_j"][2]], (1, n_t))
            @. partials[mic, "V_j"] = dnoise_dvj
            dnoise_dMj = reshape(J[i, idx["M_j"][1]:idx["M_j"][2]], (1, n_t))
            @. partials[mic, "M_j"] = dnoise_dMj
            dnoise_dAj = reshape(J[i, idx["A_j"][1]:idx["A_j"][2]], (1, n_t))
            @. partials[mic, "A_j"] = dnoise_dAj
            dnoise_dTtj = reshape(J[i, idx["Tt_j"][1]:idx["Tt_j"][2]], (1, n_t))
            @. partials[mic, "Tt_j"] = dnoise_dTtj
        elseif settings["jet_shock_source"] ==true && settings["jet_mixing_source"] == true
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
        if settings["airframe_source"]
            dnoise_dtheta_flaps = reshape(J[i, idx["theta_flaps"][1]:idx["theta_flaps"][2]], (1, n_t))
            @. partials[mic, "theta_flaps"] = dnoise_dtheta_flaps
            dnoise_dI_landing_gear = reshape(J[i, idx["I_landing_gear"][1]:idx["I_landing_gear"][2]], (1, n_t))
            @. partials[mic, "I_landing_gear"] = dnoise_dI_landing_gear
        end
    end

    println("Done computing partials noise.")

end


