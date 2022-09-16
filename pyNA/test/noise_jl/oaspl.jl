using ReverseDiff: JacobianTape, jacobian!, compile
include("../../src/noise_src_jl/oaspl.jl")

# Inputs 
x = ones(24)
y = 1.

# Compute
println("--- Compute ---")
@time f_oaspl!(y, x)
println(y)

# Compute partials
println("\n--- Compute partials ----")
Y = zeros(1)
X = zeros(24)
J = Y.*X'
#'

const f_tape = JacobianTape(f_oaspl!, Y, X)
const compiled_f_tape = compile(f_tape)

@time jacobian!(J, compiled_f_tape, x)
println(J)