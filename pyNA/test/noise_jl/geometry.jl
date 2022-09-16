using ReverseDiff: JacobianTape, jacobian!, compile
include("../../src/noise_src_jl/geometry.jl")

# Inputs 
x = 1000.
y = 0.
z = 10.
t_s = 23.
alpha = 10.
gamma = 12.
c_0 = 300.
T_0 = 288.

x = vcat(x, y, z, alpha, gamma, t_s, c_0, T_0)
y = zeros(6)

x_obs = [3500., 450., 1.2]

# Compute
println("--- Compute ---")
@time geometry_fwd!(y, x)
println(y)

# Compute partials
println("\n--- Compute partials ----")
Y = zeros(6)
X = zeros(8)
J = Y.*X'
#'

const f_tape = JacobianTape(geometry_fwd!, Y, X)
const compiled_f_tape = compile(f_tape)

@time jacobian!(J, compiled_f_tape, x)
println(J)