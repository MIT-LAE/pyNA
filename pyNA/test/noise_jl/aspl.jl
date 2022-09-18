using ReverseDiff: JacobianTape, jacobian!, compile
using CSV
using DataFrames
using BenchmarkTools
include("../../src/noise_src_jl/aspl.jl")
include("../../src/noise_src_jl/get_interpolation_functions.jl")

# Inputs 
struct Data
	aw_freq
	aw_db
end
df = Matrix(DataFrame(CSV.File("../../data/levels/weighting_a.csv")))
data = Data(df[:,1], df[:,2])

struct PynaInterpolations
	f_aw
end
f_aw = get_a_weighting_interpolation_functions(data)
pyna_ip = PynaInterpolations(f_aw)

f = [50.11,63.09,79.43,100.,125.89,158.49,199.53,251.19,316.23,398.11,501.19,630.96,794.33,1000.,1258.93,1584.89,1995.26,2511.89,3162.28,3981.07,5011.87,6309.57,7943.28,10000.]

x = rand(24)
y = ones(24)

# Compute
println("--- Compute ---")
@btime f_aspl_fwd(x)

# Compute partials
println("\n--- Compute partials ----")
Y = ones(24)
X = ones(24)
J = Y.*X'
#'

const f_tape = JacobianTape(f_aspl_fwd, X)
const compiled_f_tape = compile(f_tape)

@btime jacobian!(J, compiled_f_tape, x)
println(J)