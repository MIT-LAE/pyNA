# Imports 
using OpenMDAO: AbstractExplicitComp, VarData, PartialsData, get_rows_cols
import OpenMDAO: setup, compute!, compute_partials!
using Interpolations: LinearInterpolation
using Statistics: mean
using ForwardDiff: jacobian!, JacobianConfig#, gradient, GradientConfig
using LinearAlgebra
using ComponentArrays
using ConcreteStructs

# Define a typed settings structure
@concrete struct Levels <: AbstractExplicitComp
    n_t
    settings
    data
    levels!
    X
    Y
    J
    levels_fwd!
    levels_config

end

function Levels(n_t:: Int, settings, data)

    include(settings.pyNA_directory * "/src/noise_src_jl/spl.jl")
    include(settings.pyNA_directory * "/src/noise_src_jl/oaspl.jl")
    include(settings.pyNA_directory * "/src/noise_src_jl/pnlt.jl")
    
    # Overall levels function
    function levels!(n_t, settings, data, y, x)
        # Unpack the inputs
        rho_0 = x[:rho_0]
        c_0 = x[:c_0]
        msap_prop = x[:msap_prop]

        # Compute Levels
        spl     = f_spl(settings, msap_prop, rho_0, c_0)
        oaspl   = f_oaspl(settings, spl)
        pnlt, C = f_pnlt(settings, data, n_t, spl)

        # Pack outputs
        y[:spl] = spl
        y[:oaspl] = oaspl
        y[:pnlt] = pnlt
        if settings.bandshare
            y[:C] = C
        end
    end

    # Jacobian configuration for instantaneous noise metrics
    X = ComponentArray(rho_0 = zeros(Float64, (n_t,)),
                       c_0 = zeros(Float64, (n_t,)),
                       msap_prop = zeros(Float64, (n_t, settings.N_f)),
                       )

    if settings.bandshare
        Y = ComponentArray(spl = zeros(Float64, (n_t, settings.N_f)),
                           oaspl=zeros(Float64, (n_t,)),
                           pnlt=zeros(Float64, (n_t,)),
                           C=zeros(Float64, (n_t, settings.N_f)))
    else
        Y = ComponentArray(spl = zeros(Float64, (n_t, settings.N_f)),
                           oaspl=zeros(Float64, (n_t,)),
                           pnlt=zeros(Float64, (n_t,)))
    end

    J = Y.*X'
    #'

    levels_fwd! = (y, x)->levels!(n_t, settings, data, y, x)
    levels_config = JacobianConfig(levels_fwd!, Y, X)

    return Levels(n_t, settings, data, levels!, X, Y, J, levels_fwd!, levels_config)
end

function setup(self::Levels)
    # Load options
    n_t = self.n_t
    settings = self.settings

    # Define inputs
    inputs = Vector{VarData}()
    push!(inputs, VarData("msap_prop", shape=(n_t, settings.N_f), val=zeros((n_t, settings.N_f))))
    push!(inputs, VarData("rho_0", shape=(n_t,), val=zeros(n_t,), units="kg/m**3"))
    push!(inputs, VarData("c_0", shape=(n_t,), val=zeros(n_t,), units="m/s"))

    # Define outputs
    outputs = Vector{VarData}()
    push!(outputs, VarData("spl", shape=(n_t,settings.N_f), val=zeros((n_t, settings.N_f))))
    push!(outputs, VarData("oaspl", shape=(n_t,), val=zeros(n_t,)))
    push!(outputs, VarData("pnlt", shape=(n_t,), val=zeros(n_t,)))
    if settings.bandshare
        push!(outputs, VarData("C", shape=(n_t,settings.N_f), val=zeros((n_t, settings.N_f))))
    end

    # Define partials
    partials = Vector{PartialsData}()

    # Define sparsity pattern of the matrices
    ss_sizes = Dict(:i=>n_t, :j=>settings.N_f, :k=>1);
    rows_mv, cols_mv = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:i])  # Partials of matrix (ntxNf) wrt vector (nt)
    rows_vm, cols_vm = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:i, :j])  # Partials of vector (nt) wrt matrix (ntxNf)
    rows_vv = cols_vv = range(0,n_t-1, step=1)                               # Partials of vector (nt) wrt vector (nt)
    
    # Partials of matrix (ntxNf) wrt matrix (ntxNf) (general case)
    rows_mm = Vector{Int64}()
    cols_mm = Vector{Int64}()
    cntr = -1
    for i in range(1, n_t, step=1)
        for j in range(1, settings.N_f, step=1)
            cntr = cntr + 1
            append!(rows_mm, cntr*ones(settings.N_f))
            append!(cols_mm, range((i-1)*settings.N_f, i*settings.N_f-1, step=1))
        end
    end

    rows_mm_d, cols_mm_d = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:i, :j])  # Partials of matrix (ntxNf) wrt matrix (ntxNf) (special case: only diagonals)

    push!(partials, PartialsData("spl", "rho_0", rows=rows_mv, cols=cols_mv))
    push!(partials, PartialsData("spl", "c_0", rows=rows_mv, cols=cols_mv))
    push!(partials, PartialsData("spl", "msap_prop", rows=rows_mm_d, cols=cols_mm_d))

    push!(partials, PartialsData("oaspl", "rho_0", rows=rows_vv, cols=cols_vv))
    push!(partials, PartialsData("oaspl", "c_0", rows=rows_vv, cols=cols_vv))
    push!(partials, PartialsData("oaspl", "msap_prop", rows=rows_vm, cols=cols_vm))

    push!(partials, PartialsData("spl", "rho_0", rows=rows_mv, cols=cols_mv))
    push!(partials, PartialsData("spl", "c_0", rows=rows_mv, cols=cols_mv))
    push!(partials, PartialsData("spl", "msap_prop", rows=rows_mm_d, cols=cols_mm_d))
        
    push!(partials, PartialsData("pnlt", "rho_0", rows=rows_vv, cols=cols_vv))
    push!(partials, PartialsData("pnlt", "c_0", rows=rows_vv, cols=cols_vv))
    push!(partials, PartialsData("pnlt", "msap_prop", rows=rows_vm, cols=cols_vm))
        
    if settings.bandshare
        push!(partials, PartialsData("C", "rho_0", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("C", "c_0", rows=rows_mv, cols=cols_mv))
        push!(partials, PartialsData("C", "msap_prop", rows=rows_mm, cols=cols_mm))
    end
    
    return inputs, outputs, partials
end

function compute!(self::Levels, inputs, outputs)
    # Load options
    n_t = self.n_t
    settings = self.settings
    data = self.data
    X = self.X
    Y = self.Y

    # Print start statement
    println("Computing noise levels")
    
    # Inputs
    X[:rho_0] = inputs["rho_0"]
    X[:c_0] = inputs["c_0"]
    X[:msap_prop] = inputs["msap_prop"]

    # Initialize outputs
    Y[:spl] = zeros(Float64, (n_t, settings.N_f))
    Y[:oaspl] = zeros(Float64, (n_t,))
    Y[:pnlt] = zeros(Float64, (n_t,))
    if settings.bandshare
        Y[:C] = zeros(Float64, (n_t, settings.N_f))
    end

    # Compute levels 
    self.levels!(n_t, settings, data, Y, X)
    
    # Extract outputs
    @. outputs["spl"] = Y[:spl]
    @. outputs["oaspl"] = Y[:oaspl]
    @. outputs["pnlt"] = Y[:pnlt]

    if settings.bandshare
        @. outputs["C"] = Y[:C]
    end

end

function compute_partials!(self::Levels, inputs, partials)
    # Load options
    n_t = self.n_t
    settings = self.settings
    data = self.data
    X = self.X
    Y = self.Y
    J = self.J

    # Print start statement
    println("Computing partials noise levels")    

    # Compute partials of arrays
    X[:rho_0] = inputs["rho_0"]
    X[:c_0] = inputs["c_0"]
    X[:msap_prop] = inputs["msap_prop"]

    # Compute jacobian 
    jacobian!(J, self.levels_fwd!, Y, X, self.levels_config)

    # SPL 
    dspl_drho_0 = reshape(J[:spl, :rho_0], (settings.N_f,))
    dspl_dc_0 = reshape(J[:spl, :c_0], (settings.N_f,))
    dspl_dmsap_prop = LinearAlgebra.diagind(J[:spl, :msap_prop][1,:,1,:])

    @. partials["spl", "rho_0"] = dspl_drho_0
    @. partials["spl", "c_0"]   = dspl_dc_0
    @. partials["spl", "msap_prop"] = dspl_dmsap_prop


    #@. partials["spl", "rho_0"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:spl, :rho_0]
    #@. partials["spl", "c_0"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:spl, :c_0]
    #dspl_dmsap_prop = diag(J[:spl, :msap_prop])
    #@. partials["spl", "msap_prop"][(i-1)*settings.N_f+1:i*settings.N_f] = dspl_dmsap_prop

        ## Partials of oaspl
        #partials["oaspl", "rho_0"][i] = J[:oaspl, :rho_0]
        #partials["oaspl", "c_0"][i] = J[:oaspl, :c_0]
        #@. partials["oaspl", "msap_prop"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:oaspl, :msap_prop]

        ## Compute pnlt
        #@. partials["pnlt", "rho_0"] = J[:pnlt, :rho_0]
        #@. partials["pnlt", "c_0"] = J[:pnlt, :c_0]
        #@. partials["pnlt", "msap_prop"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:pnlt, :msap_prop]
        
        ## Compute C
        #if settings.bandshare
        #    @. partials["C", "rho_0"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:C, :rho_0]
        #    @. partials["C", "c_0"][(i-1)*settings.N_f+1:i*settings.N_f] = J[:C, :c_0]
        #    for j in range(1, settings.N_f, step=1)
        #        dC_dmsap_prop = @view(J[:C, :msap_prop][:,j])
        #        @. partials["C", "msap_prop"][(i-1)*settings.N_f+1:i*settings.N_f] = dC_dmsap_prop
        #    end
        #end

    #end
end

