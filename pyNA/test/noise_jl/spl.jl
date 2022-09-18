using ReverseDiff: JacobianTape, jacobian!, compile
using BenchmarkTools
include("../../src/noise_src_jl/spl.jl")

# Inputs 
c_0 = 300
rho_0 = 1.2

x = vcat(c_0, rho_0)
y = 100*ones(24)

# Compute
println("--- Compute ---")
@btime f_spl!(y, x)

# Compute partials
println("\n--- Compute partials ----")
Y = 100*ones(24)
X = vcat(c_0, rho_0)
J = Y.*X'
#'

const f_tape = JacobianTape(f_spl!, Y, X)
const compiled_f_tape = compile(f_tape)

@btime jacobian!(J, compiled_f_tape, x)
println(J)