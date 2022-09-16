using ReverseDiff: JacobianTape, jacobian!, compile
include("../../src/noise_src_jl/smooth_max.jl")

# Inputs 
x = -((1:1:10) .-5).^2 .+ rand()
y = zeros(1)
k_smooth = 50

# Compute
println("--- Compute ---")
@time smooth_max_fwd!(y, x)
println(y)

# Compute partials
println("\n--- Compute partials ----")
Y = zeros(1)
X = zeros(10)
J = Y.*X'
#'

const f_tape = JacobianTape(smooth_max_fwd!, Y, X)
const compiled_f_tape = compile(f_tape)

@time jacobian!(J, compiled_f_tape, x)
println(J)