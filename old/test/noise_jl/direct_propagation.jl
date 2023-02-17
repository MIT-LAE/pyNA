using ReverseDiff: JacobianTape, jacobian!, compile
include("../../src/noise_src_jl/direct_propagation.jl")

# Inputs 
r = 100.
I_0 = 400.

x = vcat(r, I_0)
y = ones(24)

settings = Dict()
settings["r_0"] = 0.3048

# Compute
println("--- Compute ---")
@time direct_propagation_fwd!(y, x)
println(y)

# Compute partials
println("\n--- Compute partials ----")
Y = ones(24)
X = ones(26)
J = Y.*X'
#'

const f_tape = JacobianTape(direct_propagation_fwd!, Y, X)
const compiled_f_tape = compile(f_tape)

@time jacobian!(J, compiled_f_tape, x)
println(J)