# Imports 
using OpenMDAO: AbstractExplicitComp, VarData, PartialsData, get_rows_cols
import OpenMDAO: setup, compute!, compute_partials!
using Interpolations: LinearInterpolation
using PCHIPInterpolation
using Statistics: mean
using ForwardDiff: jacobian!, JacobianConfig
using ComponentArrays
using ConcreteStructs

# Define source struct
@concrete struct Source{} <: AbstractExplicitComp
    settings
    data
    ac
    shield
    n_t
    idx_src
    source!
    X
    Y
    J
end

function Source(settings, data, ac, shield, n_t::Int, idx_src)
    
    include(settings.pyNA_directory * "/src/noise_src_jl/fan.jl")

    include(settings.pyNA_directory * "/src/noise_src_jl/core.jl")

    include(settings.pyNA_directory * "/src/noise_src_jl/jet.jl")

    include(settings.pyNA_directory * "/src/noise_src_jl/airframe.jl")

    function source!(settings, data, ac, shield, n_t, idx_src, y, x)

        # Create input_src
        input_src = x[:TS]
        input_src = vcat(input_src, x[:M_0])
        input_src = vcat(input_src, x[:c_0])
        input_src = vcat(input_src, x[:rho_0])
        input_src = vcat(input_src, x[:mu_0])
        input_src = vcat(input_src, x[:T_0])
        input_src = vcat(input_src, x[:theta])
        input_src = vcat(input_src, x[:phi])
        if settings.jet_mixing == true && settings.jet_shock == false
            input_src = vcat(input_src, x[:V_j_star])
            input_src = vcat(input_src, x[:rho_j_star])
            input_src = vcat(input_src, x[:A_j_star])
            input_src = vcat(input_src, x[:Tt_j_star])
        elseif settings.jet_shock == true && settings.jet_mixing == false
            input_src = vcat(input_src, x[:V_j_star])
            input_src = vcat(input_src, x[:M_j])
            input_src = vcat(input_src, x[:A_j_star])
            input_src = vcat(input_src, x[:Tt_j_star])
        elseif settings.jet_shock ==true && settings.jet_mixing == true
            input_src = vcat(input_src, x[:V_j_star])
            input_src = vcat(input_src, x[:rho_j_star])
            input_src = vcat(input_src, x[:A_j_star])
            input_src = vcat(input_src, x[:Tt_j_star])
            input_src = vcat(input_src, x[:M_j])
        end
        if settings.core
            if settings.method_core_turb == "GE"
                input_src = vcat(input_src, x[:mdoti_c_star])
                input_src = vcat(input_src, x[:Tti_c_star])
                input_src = vcat(input_src, x[:Ttj_c_star])
                input_src = vcat(input_src, x[:Pti_c_star])
                input_src = vcat(input_src, x[:DTt_des_c_star])
            elseif settings.method_core_turb == "PW"
                input_src = vcat(input_src, x[:mdoti_c_star])
                input_src = vcat(input_src, x[:Tti_c_star])
                input_src = vcat(input_src, x[:Ttj_c_star])
                input_src = vcat(input_src, x[:Pti_c_star])
                input_src = vcat(input_src, x[:rho_te_c_star])
                input_src = vcat(input_src, x[:c_te_c_star])
                input_src = vcat(input_src, x[:rho_ti_c_star])
                input_src = vcat(input_src, x[:c_ti_c_star])
            end
        end
        if settings.airframe
            input_src = vcat(input_src, x[:theta_flaps])
            input_src = vcat(input_src, x[:I_landing_gear])
        end
        if settings.fan_inlet==true || settings.fan_discharge==true
            input_src = vcat(input_src, x[:DTt_f_star])
            input_src = vcat(input_src, x[:mdot_f_star])
            input_src = vcat(input_src, x[:N_f_star])
            input_src = vcat(input_src, x[:A_f_star])
            input_src = vcat(input_src, x[:d_f_star])
        end

        # Extract TS
        TS = x[:TS]

        # Get type of input vector
        T = eltype(input_src)

        # Initialize source mean-square acoustic pressure
        msap_source = zeros(T, (n_t, settings.N_f))

        if settings.fan_inlet
            msap_fan_inlet = fan(settings, data, ac, n_t, shield, idx_src, input_src, "fan_inlet")
            msap_source = msap_source .+ msap_fan_inlet
        end
        if settings.fan_discharge
            msap_fan_discharge = fan(settings, data, ac, n_t, shield, idx_src, input_src, "fan_discharge")
            msap_source = msap_source .+ msap_fan_discharge
        end

        if settings.core
            msap_core = core(settings, data, ac, n_t, idx_src, input_src)
            if settings.suppression && settings.case_name in ["NASA STCA Standard", "stca_enginedesign_standard"]
                msap_core[findall(TS.*ones(1, settings.N_f).>0.8)] = (10. ^(-2.3 / 10.) * msap_core)[findall(TS.*ones(1, settings.N_f).>0.8)]
            end
            msap_source = msap_source .+ msap_core
        end

        if settings.jet_mixing 
            msap_jet_mixing = jet_mixing(settings, data, ac, n_t, idx_src, input_src)
            if settings.suppression && settings.case_name in ["NASA STCA Standard", "stca_enginedesign_standard"]
                msap_jet_mixing[findall(TS.*ones(1, settings.N_f).>0.8)] = (10. ^(-2.3 / 10.) * msap_jet_mixing)[findall(TS.*ones(1, settings.N_f).>0.8)]
            end
            msap_source = msap_source .+ msap_jet_mixing
        end
        if settings.jet_shock
            msap_jet_shock = jet_shock(settings, data, ac, n_t, idx_src, input_src)
            if settings.suppression && settings.case_name in ["NASA STCA Standard", "stca_enginedesign_standard"]
                msap_jet_shock[findall(TS.*ones(1, settings.N_f).>0.8)] = (10. ^(-2.3 / 10.) * msap_jet_shock)[findall(TS.*ones(1, settings.N_f).>0.8)]
            end
            msap_source = msap_source .+ msap_jet_shock
        end

        if settings.airframe
            msap_af = airframe(settings, data, ac, n_t, idx_src, input_src)
            msap_source = msap_source .+ msap_af
        end

         y[:msap_source] = 10*log10.(msap_source)
        #y[:msap_source] = msap_source
        
    end

    X = ComponentArray(TS=1.,
                       M_0=1.,
                       c_0=1.,
                       rho_0=1.,
                       mu_0=1.,
                       T_0=1.,
                       theta=1.,
                       phi=1.)
    if settings.jet_mixing == true && settings.jet_shock == false
        X = vcat(X, ComponentArray(V_j_star=1.,
                                   rho_j_star=1.,
                                   A_j_star=1.,
                                   Tt_j_star=1.))
    elseif settings.jet_shock == true && settings.jet_mixing == false
        X = vcat(X, ComponentArray(V_j_star=1.,
                                   M_j=1.,
                                   A_j_star=1.,
                                   Tt_j_star=1.))
    elseif settings.jet_shock ==true && settings.jet_mixing == true
        X = vcat(X, ComponentArray(V_j_star=1.,
                                   rho_j_star=1.,
                                   A_j_star=1.,
                                   Tt_j_star=1.,
                                   M_j=1.))
    end
    if settings.core
        if settings.method_core_turb == "GE"
            X = vcat(X, ComponentArray(mdoti_c_star=1.,
                                       Tti_c_star=1.,
                                       Ttj_c_star=1.,
                                       Pti_c_star=1.,
                                       DTt_des_c_star=1.))
        elseif settings.method_core_turb == "PW"
            X = vcat(X, ComponentArray(mdoti_c_star=1.,
                                       Tti_c_star=1.,
                                       Ttj_c_star=1.,
                                       Pti_c_star=1.,
                                       rho_te_c_star=1.,
                                       c_te_c_star=1.,
                                       rho_ti_c_star=1.,
                                       c_ti_c_star=1.))
        end
    end
    if settings.airframe
        X = vcat(X, ComponentArray(theta_flaps=1.,
                                   I_landing_gear=1.))
    end
    if settings.fan_inlet==true || settings.fan_discharge==true
        X = vcat(X, ComponentArray(DTt_f_star=1.,
                                   mdot_f_star=1.,
                                   N_f_star=1.,
                                   A_f_star=1.,
                                   d_f_star=1.))
    end

    Y = ComponentArray(msap_source=zeros(Float64, settings.N_f))
    J = Y.*X'
    #'

    return Source(settings, data, ac, shield, n_t, idx_src, source!, X, Y, J)
end

function setup(self::Source)
    # Load options
    n_t = self.n_t
    settings = self.settings

    # Create inputs
    inputs = Vector{VarData}()
    if settings.jet_mixing == true && settings.jet_shock == false
        push!(inputs, VarData("V_j_star", shape=(n_t,), val=ones(n_t)))
        push!(inputs, VarData("rho_j_star", shape=(n_t,), val=ones(n_t)))
        push!(inputs, VarData("A_j_star", shape=(n_t,), val=ones(n_t)))
        push!(inputs, VarData("Tt_j_star", shape=(n_t,), val=ones(n_t)))
    elseif settings.jet_shock == true && settings.jet_mixing == false
        push!(inputs, VarData("V_j_star", shape=(n_t,), val=ones(n_t)))
        push!(inputs, VarData("M_j", shape=(n_t,), val=ones(n_t)))
        push!(inputs, VarData("A_j_star", shape=(n_t,), val=ones(n_t)))
        push!(inputs, VarData("Tt_j_star", shape=(n_t,), val=ones(n_t)))
    elseif settings.jet_shock ==true && settings.jet_mixing == true
        push!(inputs, VarData("V_j_star", shape=(n_t,), val=ones(n_t)))
        push!(inputs, VarData("rho_j_star", shape=(n_t,), val=ones(n_t)))
        push!(inputs, VarData("A_j_star", shape=(n_t,), val=ones(n_t)))
        push!(inputs, VarData("Tt_j_star", shape=(n_t,), val=ones(n_t)))
        push!(inputs, VarData("M_j", shape=(n_t,), val=ones(n_t)))
    end
    if settings.core
        if settings.method_core_turb == "GE"
            push!(inputs, VarData("mdoti_c_star", shape=(n_t,), val=ones(n_t)))
            push!(inputs, VarData("Tti_c_star", shape=(n_t,), val=ones(n_t)))
            push!(inputs, VarData("Ttj_c_star", shape=(n_t,), val=ones(n_t)))
            push!(inputs, VarData("Pti_c_star", shape=(n_t,), val=ones(n_t)))
            push!(inputs, VarData("DTt_des_c_star", shape=(n_t,), val=ones(n_t)))
        elseif settings.method_core_turb == "PW"
            push!(inputs, VarData("mdoti_c_star", shape=(n_t,), val=ones(n_t)))
            push!(inputs, VarData("Tti_c_star", shape=(n_t,), val=ones(n_t)))
            push!(inputs, VarData("Ttj_c_star", shape=(n_t,), val=ones(n_t)))
            push!(inputs, VarData("Pti_c_star", shape=(n_t,), val=ones(n_t)))
            push!(inputs, VarData("rho_te_c_star", shape=(n_t,), val=ones(n_t)))
            push!(inputs, VarData("c_te_c_star", shape=(n_t,), val=ones(n_t)))
            push!(inputs, VarData("rho_ti_c_star", shape=(n_t,), val=ones(n_t)))
            push!(inputs, VarData("c_ti_c_star", shape=(n_t,), val=ones(n_t)))
        end
    end
    if settings.airframe
        push!(inputs, VarData("theta_flaps", shape=(n_t,), val=ones(n_t), units="deg"))
        push!(inputs, VarData("I_landing_gear", shape=(n_t,), val=ones(n_t)))
    end
    if settings.fan_inlet==true || settings.fan_discharge==true
        push!(inputs, VarData("DTt_f_star", shape=(n_t,), val=ones(n_t)))
        push!(inputs, VarData("mdot_f_star", shape=(n_t,), val=ones(n_t)))
        push!(inputs, VarData("N_f_star", shape=(n_t,), val=ones(n_t)))
        push!(inputs, VarData("A_f_star", shape=(n_t,), val=ones(n_t)))
        push!(inputs, VarData("d_f_star", shape=(n_t,), val=ones(n_t)))
    end

    # Power setting input
    push!(inputs, VarData("TS", shape=(n_t,), val=ones(n_t)))
    push!(inputs, VarData("M_0", shape=(n_t,), val=ones(n_t)))
    push!(inputs, VarData("c_0", shape=(n_t,), val=ones(n_t), units="m/s"))
    push!(inputs, VarData("rho_0", shape=(n_t,), val=ones(n_t), units="kg/m**3"))
    push!(inputs, VarData("mu_0", shape=(n_t,), val=ones(n_t), units="kg/m/s"))
    push!(inputs, VarData("T_0", shape=(n_t,), val=ones(n_t), units="K"))
    push!(inputs, VarData("theta", shape=(n_t,), val=ones(n_t), units="deg"))
    push!(inputs, VarData("phi", shape=(n_t,), val=ones(n_t), units="deg"))

    # Create outputs
    # Output
    outputs = Vector{VarData}()
    push!(outputs, VarData("msap_source", shape=((n_t, settings.N_f)), val=zeros((n_t, settings.N_f))))

    # Create partials 
    partials = Vector{PartialsData}()
    ss_sizes = Dict(:i=>n_t, :j=>settings.N_f, :k=>1);
    rows_mv, cols_mv = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:i])  # Partials of matrix (ntxNf) wrt vector (nt)
    if settings.jet_mixing && settings.jet_shock == false
        push!(partials, PartialsData("msap_source", "V_j_star", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("msap_source", "rho_j_star", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("msap_source", "A_j_star", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("msap_source", "Tt_j_star", rows=rows_mv, cols=cols_mv))
    elseif settings.jet_shock && settings.jet_mixing == false
        push!(partials, PartialsData("msap_source", "V_j_star", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("msap_source", "A_j_star", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("msap_source", "Tt_j_star", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("msap_source", "M_j", rows=rows_mv, cols=cols_mv))
    elseif settings.jet_shock && settings.jet_mixing
        push!(partials, PartialsData("msap_source", "V_j_star", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("msap_source", "rho_j_star", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("msap_source", "A_j_star", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("msap_source", "Tt_j_star", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("msap_source", "M_j", rows=rows_mv, cols=cols_mv))
    end

    if settings.core
        if settings.method_core_turb == "GE"
            push!(partials, PartialsData("msap_source", "mdoti_c_star", rows=rows_mv, cols=cols_mv))
            push!(partials, PartialsData("msap_source", "Tti_c_star", rows=rows_mv, cols=cols_mv))
            push!(partials, PartialsData("msap_source", "Ttj_c_star", rows=rows_mv, cols=cols_mv))
            push!(partials, PartialsData("msap_source", "Pti_c_star", rows=rows_mv, cols=cols_mv))
            push!(partials, PartialsData("msap_source", "DTt_des_c_star", rows=rows_mv, cols=cols_mv))
        elseif settings.method_core_turb == "PW"
            push!(partials, PartialsData("msap_source", "mdoti_c_star", rows=rows_mv, cols=cols_mv))
            push!(partials, PartialsData("msap_source", "Tti_c_star", rows=rows_mv, cols=cols_mv))
            push!(partials, PartialsData("msap_source", "Ttj_c_star", rows=rows_mv, cols=cols_mv))
            push!(partials, PartialsData("msap_source", "Pti_c_star", rows=rows_mv, cols=cols_mv))
            push!(partials, PartialsData("msap_source", "rho_te_c", rows=rows_mv, cols=cols_mv))
            push!(partials, PartialsData("msap_source", "c_te_c", rows=rows_mv, cols=cols_mv))
            push!(partials, PartialsData("msap_source", "rho_ti_c", rows=rows_mv, cols=cols_mv))
            push!(partials, PartialsData("msap_source", "c_ti_c", rows=rows_mv, cols=cols_mv))
        end
    end
    if settings.fan_inlet==true || settings.fan_discharge==true
        push!(partials, PartialsData("msap_source", "DTt_f_star", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("msap_source", "mdot_f_star", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("msap_source", "N_f_star", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("msap_source", "A_f_star", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("msap_source", "d_f_star", rows=rows_mv, cols=cols_mv))
    end
    if settings.airframe
        push!(partials, PartialsData("msap_source", "theta_flaps", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("msap_source", "I_landing_gear", rows=rows_mv, cols=cols_mv))
    end

    push!(partials, PartialsData("msap_source", "theta", rows=rows_mv, cols=cols_mv))
    push!(partials, PartialsData("msap_source", "phi", rows=rows_mv, cols=cols_mv))
    push!(partials, PartialsData("msap_source", "TS", rows=rows_mv, cols=cols_mv))
    push!(partials, PartialsData("msap_source", "M_0", rows=rows_mv, cols=cols_mv))
    push!(partials, PartialsData("msap_source", "c_0", rows=rows_mv, cols=cols_mv))
    push!(partials, PartialsData("msap_source", "mu_0", rows=rows_mv, cols=cols_mv))
    push!(partials, PartialsData("msap_source", "rho_0", rows=rows_mv, cols=cols_mv))
    push!(partials, PartialsData("msap_source", "T_0", rows=rows_mv, cols=cols_mv))

    return inputs, outputs, partials
end

function compute!(self::Source, inputs, outputs)
    # Loda options
    settings = self.settings
    data = self.data
    ac = self.ac
    shield = self.shield
    n_t = self.n_t
    idx_src = self.idx_src
    X = self.X
    Y = self.Y

    # Print start statement
    println("Computing noise sources")

    for i  in range(1, n_t, step=1)
        if settings.fan_inlet
            X[:DTt_f_star] = inputs["DTt_f_star"][i]
            X[:mdot_f_star] = inputs["mdot_f_star"][i]
            X[:N_f_star] = inputs["N_f_star"][i]
            X[:A_f_star] = inputs["A_f_star"][i]
            X[:d_f_star] = inputs["d_f_star"][i]
            X[:M_0] = inputs["M_0"][i]
            X[:c_0] = inputs["c_0"][i]
            X[:T_0] = inputs["T_0"][i]
            X[:rho_0] = inputs["rho_0"][i]
            X[:theta] = inputs["theta"][i]
        end
        if settings.fan_discharge
            X[:DTt_f_star] = inputs["DTt_f_star"][i]
            X[:mdot_f_star] = inputs["mdot_f_star"][i]
            X[:N_f_star] = inputs["N_f_star"][i]
            X[:A_f_star] = inputs["A_f_star"][i]
            X[:d_f_star] = inputs["d_f_star"][i]
            X[:M_0] = inputs["M_0"][i]
            X[:c_0] = inputs["c_0"][i]
            X[:T_0] = inputs["T_0"][i]
            X[:rho_0] = inputs["rho_0"][i]
            X[:theta] = inputs["theta"][i]
        end

        if settings.core
            if settings.method_core_turb == "GE"
                X[:mdoti_c_star] = inputs["mdoti_c_star"][i]
                X[:Tti_c_star] = inputs["Tti_c_star"][i]
                X[:Ttj_c_star] = inputs["Ttj_c_star"][i]
                X[:Pti_c_star] = inputs["Pti_c_star"][i]
                X[:DTt_des_c_star] = inputs["DTt_des_c_star"][i]
                X[:TS] = inputs["TS"][i]
                X[:theta] = inputs["theta"][i]
                X[:M_0] = inputs["M_0"][i]
            elseif settings.method_core_turb == "PW"
                X[:mdoti_c_star] = inputs["mdoti_c_star"][i]
                X[:Tti_c_star] = inputs["Tti_c_star"][i]
                X[:Ttj_c_star] = inputs["Ttj_c_star"][i]
                X[:Pti_c_star] = inputs["Pti_c_star"][i]
                X[:rho_te_c_star] = inputs["rho_te_c_star"][i]
                X[:c_te_c_star] = inputs["c_te_c_star"][i]
                X[:rho_ti_c_star] = inputs["rho_ti_c_star"][i]
                X[:c_ti_c_star] = inputs["c_ti_c_star"][i]
                X[:TS] = inputs["TS"][i]
                X[:theta] = inputs["theta"][i]
                X[:M_0] = inputs["M_0"][i]
            else
                throw(DomainError("Invalid method for turbine noise in core module. Specify: GE/PW."))
            end
        end

        if settings.jet_mixing 
            X[:V_j_star] = inputs["V_j_star"][i]
            X[:rho_j_star] = inputs["rho_j_star"][i]
            X[:A_j_star] = inputs["A_j_star"][i]
            X[:Tt_j_star] = inputs["Tt_j_star"][i]
            X[:TS] = inputs["TS"][i]
            X[:M_0] = inputs["M_0"][i]
            X[:c_0] = inputs["c_0"][i]
            X[:theta] = inputs["theta"][i]
        end
        if settings.jet_shock
            X[:V_j_star] = inputs["V_j_star"][i]
            X[:M_j] = inputs["M_j"][i]
            X[:A_j_star] = inputs["A_j_star"][i]
            X[:Tt_j_star] = inputs["Tt_j_star"][i]
            X[:TS] = inputs["TS"][i]
            X[:M_0] = inputs["M_0"][i]
            X[:c_0] = inputs["c_0"][i]
            X[:theta] = inputs["theta"][i]
        end

        if settings.airframe
            X[:theta_flaps] = inputs["theta_flaps"][i]
            X[:I_landing_gear] = round(inputs["I_landing_gear"][i])
            X[:M_0] = inputs["M_0"][i]
            X[:c_0] = inputs["c_0"][i]
            X[:rho_0] = inputs["rho_0"][i]
            X[:mu_0] = inputs["mu_0"][i]
            X[:theta] = inputs["theta"][i]
            X[:phi] = inputs["phi"][i]
        end

        Y[:msap_source] = zeros(Float64, settings.N_f)

        self.source!(settings, data, ac, shield[i, 2:end], n_t, idx_src, Y, X)
        msap_source = Y[:msap_source]
        @. outputs["msap_source"][i,:] = msap_source 

    end
end

function compute_partials!(self::Source, inputs, partials)
    # Load options
    n_t = self.n_t
    settings = self.settings
    data = self.data
    ac = self.ac
    shield = self.shield
    n_t = self.n_t
    idx_src = self.idx_src
    X = self.X
    Y = self.Y 
    J = self.J

    # Print start statement
    println("Computing partials source")

    for i  in range(1, n_t, step=1)

        X[:M_0] = inputs["M_0"][i]
        X[:c_0] = inputs["c_0"][i]
        X[:T_0] = inputs["T_0"][i]
        X[:rho_0] = inputs["rho_0"][i]
        X[:mu_0] = inputs["mu_0"][i]
        X[:theta] = inputs["theta"][i]
        X[:phi] = inputs["phi"][i]
        X[:TS] = inputs["TS"][i]

        if settings.fan_inlet
            X[:DTt_f_star] = inputs["DTt_f_star"][i]
            X[:mdot_f_star] = inputs["mdot_f_star"][i]
            X[:N_f_star] = inputs["N_f_star"][i]
            X[:A_f_star] = inputs["A_f_star"][i]
            X[:d_f_star] = inputs["d_f_star"][i]
        end
        if settings.fan_discharge
            X[:DTt_f_star] = inputs["DTt_f_star"][i]
            X[:mdot_f_star] = inputs["mdot_f_star"][i]
            X[:N_f_star] = inputs["N_f_star"][i]
            X[:A_f_star] = inputs["A_f_star"][i]
            X[:d_f_star] = inputs["d_f_star"][i]
        end
        if settings.core
            if settings.method_core_turb == "GE"
                X[:mdoti_c_star] = inputs["mdoti_c_star"][i]
                X[:Tti_c_star] = inputs["Tti_c_star"][i]
                X[:Ttj_c_star] = inputs["Ttj_c_star"][i]
                X[:Pti_c_star] = inputs["Pti_c_star"][i]
                X[:DTt_des_c_star] = inputs["DTt_des_c_star"][i]
            elseif settings.method_core_turb == "PW"
                X[:mdoti_c_star] = inputs["mdoti_c_star"][i]
                X[:Tti_c_star] = inputs["Tti_c_star"][i]
                X[:Ttj_c_star] = inputs["Ttj_c_star"][i]
                X[:Pti_c_star] = inputs["Pti_c_star"][i]
                X[:rho_te_c_star] = inputs["rho_te_c_star"][i]
                X[:c_te_c_star] = inputs["c_te_c_star"][i]
                X[:rho_ti_c_star] = inputs["rho_ti_c_star"][i]
                X[:c_ti_c_star] = inputs["c_ti_c_star"][i]
            else
                throw(DomainError("Invalid method for turbine noise in core module. Specify: GE/PW."))
            end
        end

        if settings.jet_mixing 
            X[:V_j_star] = inputs["V_j_star"][i]
            X[:rho_j_star] = inputs["rho_j_star"][i]
            X[:A_j_star] = inputs["A_j_star"][i]
            X[:Tt_j_star] = inputs["Tt_j_star"][i]
        end
        if settings.jet_shock
            X[:V_j_star] = inputs["V_j_star"][i]
            X[:M_j] = inputs["M_j"][i]
            X[:A_j_star] = inputs["A_j_star"][i]
            X[:Tt_j_star] = inputs["Tt_j_star"][i]
        end

        if settings.airframe
            X[:theta_flaps] = inputs["theta_flaps"][i]
            X[:I_landing_gear] = inputs["I_landing_gear"][i]
        end

        Y[:msap_source] = zeros(Float64, settings.N_f)
        source_fwd! = (y,x)->self.source!(settings, data, ac, shield[i, 2:end], n_t, idx_src, y, x)
        jacobian!(J, source_fwd!, Y, X)

        @. partials["msap_source", "TS"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :TS]
        @. partials["msap_source", "M_0"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :M_0]
        @. partials["msap_source", "c_0"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :c_0]
        @. partials["msap_source", "rho_0"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :rho_0]
        @. partials["msap_source", "mu_0"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :mu_0]
        @. partials["msap_source", "T_0"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :T_0]
        @. partials["msap_source", "theta"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :theta]
        @. partials["msap_source", "phi"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :phi]

        if settings.fan_inlet
            @. partials["msap_source", "DTt_f_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :DTt_f_star]
            @. partials["msap_source", "mdot_f_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :mdot_f_star]
            @. partials["msap_source", "N_f_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :N_f_star]
            @. partials["msap_source", "A_f_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :A_f_star]
            @. partials["msap_source", "d_f_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :d_f_star]
        end

        if settings.fan_discharge
            @. partials["msap_source", "DTt_f_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :DTt_f_star]
            @. partials["msap_source", "mdot_f_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :mdot_f_star]
            @. partials["msap_source", "N_f_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :N_f_star]
            @. partials["msap_source", "A_f_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :A_f_star]
            @. partials["msap_source", "d_f_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :d_f_star]
        end

        if settings.core
            if settings.method_core_turb == "GE"
                @. partials["msap_source", "mdoti_c_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :mdoti_c_star]
                @. partials["msap_source", "Tti_c_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :Tti_c_star]
                @. partials["msap_source", "Ttj_c_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :Ttj_c_star]
                @. partials["msap_source", "Pti_c_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :Pti_c_star]
                @. partials["msap_source", "DTt_des_c_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :DTt_des_c_star]
                
            elseif settings.method_core_turb == "GE"
                @. partials["msap_source", "mdoti_c_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :mdoti_c_star]
                @. partials["msap_source", "Tti_c_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :Tti_c_star]
                @. partials["msap_source", "Ttj_c_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :Ttj_c_star]
                @. partials["msap_source", "Pti_c_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :Pti_c_star]
                @. partials["msap_source", "rho_te_c_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :rho_te_c_star]
                @. partials["msap_source", "c_te_c_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :c_te_c_star]
                @. partials["msap_source", "rho_ti_c_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :rho_ti_c_star]
                @. partials["msap_source", "c_ti_c_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :c_ti_c_star]
            else
                throw(DomainError("Invalid method for turbine noise in core module. Specify: GE/PW."))
            end
        end

        if settings.jet_mixing == true && settings.jet_shock == false
            @. partials["msap_source", "V_j_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :V_j_star]
            @. partials["msap_source", "rho_j_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :rho_j_star]
            @. partials["msap_source", "A_j_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :A_j_star]
            @. partials["msap_source", "Tt_j_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :Tt_j_star]
        elseif settings.jet_shock == true && settings.jet_mixing == false
            @. partials["msap_source", "V_j_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :V_j_star]
            @. partials["msap_source", "M_j"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :M_j]
            @. partials["msap_source", "A_j_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :A_j_star]
            @. partials["msap_source", "Tt_j_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :Tt_j_star]
        elseif settings.jet_shock ==true && settings.jet_mixing == true
            @. partials["msap_source", "V_j_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :V_j_star]
            @. partials["msap_source", "rho_j_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :rho_j_star]
            @. partials["msap_source", "A_j_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :A_j_star]
            @. partials["msap_source", "Tt_j_star"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :Tt_j_star]
            @. partials["msap_source", "M_j"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :M_j]
        end

        if settings.airframe
            @. partials["msap_source", "theta_flaps"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :theta_flaps]
            @. partials["msap_source", "I_landing_gear"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:msap_source, :I_landing_gear]
        end
    end
end
