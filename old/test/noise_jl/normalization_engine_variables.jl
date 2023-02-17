using ReverseDiff: JacobianTape, jacobian!, compile
include("../../src/noise_src_jl/normalization_engine_variables.jl")

# Inputs 
msap = 100*ones(24)
c_0 = 300
rho_0 = 1.2

V_j = 1000.
rho_j = 0.7
A_j = 1.
Tt_j = 500.
c_0 = 300.
rho_0 = 1.2
T_0 = 290.

settings = Dict()
settings["A_e"] = 10.

x = vcat(V_j, rho_j, A_j, Tt_j, c_0, rho_0, T_0)
y = zeros(4)

# Compute
println("--- Compute ---")
@time normalization_jet_mixing_fwd!(y, x)
println(y)

# Compute partials
println("\n--- Compute partials ----")
Y = zeros(4)
X = zeros(7)
J = Y.*X'
#'

const f_tape = JacobianTape(normalization_jet_mixing_fwd!, Y, X)
const compiled_f_tape = compile(f_tape)

@time jacobian!(J, compiled_f_tape, x)
println(J)