using ReverseDiff: JacobianTape, jacobian!, compile
include("../../src/noise_src_jl/epnl.jl")

# Inputs 
t_o = range(1,100, length=100)
level = range(40, 100, length=100)

x = vcat(t_o, level)
y = zeros(1)

settings = Dict()
settings["epnl_dt"] = 0.5

# Compute
println("--- Compute ---")
@time f_epnl_fwd!(y, x)
println(y)

# Compute partials
println("\n--- Compute partials ----")
Y = zeros(1)
X = vcat(range(1,100,length=100), range(1,100,length=100))
J = Y.*X'
#'

const f_tape = JacobianTape(f_epnl_fwd!, Y, X)
const compiled_f_tape = compile(f_tape)

@time jacobian!(J, compiled_f_tape, x)
println(J)