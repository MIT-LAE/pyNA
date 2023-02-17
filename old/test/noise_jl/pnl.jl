using ReverseDiff: JacobianTape, jacobian!, compile
include("../../src/noise_src_jl/pnl.jl")

# Inputs 
x = rand(24)
y = zeros(24)

# Compute
println("--- Compute ---")
@time f_pnl!(y, x)
println(y)

# Compute partials
println("\n--- Compute partials ----")
Y = zeros(24)
X = zeros(24)
J = Y.*X'
#'

const f_tape = JacobianTape(f_pnl!, Y, X)
const compiled_f_tape = compile(f_tape)

@time jacobian!(J, compiled_f_tape, x)
println(J)