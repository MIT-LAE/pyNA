using ReverseDiff: JacobianTape, jacobian!, compile
include("../../src/noise_src_jl/split_subbands.jl")

# Inputs 
x = [30., 40., 45., 51., 55., 53., 60., 67., 78., 89., 88., 90., 91., 90., 89., 86., 85., 83., 84., 86., 81., 77., 75., 73.]
y = zeros(120)

settings = Dict()
settings["n_frequency_bands"] = 24
settings["n_frequency_subbands"] = 5

# Compute
println("--- Compute ---")
@time split_subbands_fwd!(y, x)
println(y)

# Compute partials
println("\n--- Compute partials ----")
Y = zeros(120)
X = zeros(24)
J = Y.*X'
#'

const f_tape = JacobianTape(split_subbands_fwd!, Y, X)
const compiled_f_tape = compile(f_tape)

@time jacobian!(J, compiled_f_tape, x)
println(J)