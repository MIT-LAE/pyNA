using ReverseDiff: JacobianTape, jacobian!, compile
using BenchmarkTools
include("../../src/noise_src_jl/lateral_attenuation.jl")

# Inputs 
beta = 20.
x_obs = [3500., 450., 1.2]

x = vcat(range(0, 10000, length=101), zeros(101), range(0, 100, length=101))

settings = Dict()
settings["lateral_attenuation_engine_mounting"] = "underwing"

# Compute
println("--- Compute ---")
@btime lateral_attenuation_fwd(x)

# Compute partials
println("\n--- Compute partials ----")
Y = ones(1)
X = vcat(range(0, 10000, length=101), zeros(101), range(0, 100, length=101))
J = Y.*X'
#'

const f_tape = JacobianTape(lateral_attenuation_fwd, X)
const compiled_f_tape = compile(f_tape)

@btime jacobian!(J, compiled_f_tape, x)
println(J)