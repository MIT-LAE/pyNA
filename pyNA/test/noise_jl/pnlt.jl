using ReverseDiff: JacobianTape, jacobian!, compile
using CSV
using DataFrames
include("../../src/noise_src_jl/pnlt.jl")
include("../../src/noise_src_jl/get_interpolation_functions.jl")


# Inputs 
struct Data
	noy_spl
	noy_freq
	noy
end
df = Matrix(DataFrame(CSV.File("../../data/levels/spl_noy.csv")))
data = Data(df[2:end, 1], df[1, 2:end], df[2:end, 2:end])

struct PynaInterpolations
	f_noy
end
f_noy = get_noy_interpolation_functions(data)
pyna_ip = PynaInterpolations(f_noy)

x = [0., 0., 70., 62., 70., 80., 82., 83., 76., 80., 80., 79., 78., 80., 78., 76., 79., 85., 79., 78., 71., 60., 54., 45.]
y = zeros(1)

f = [50.11,63.09,79.43,100.,125.89,158.49,199.53,251.19,316.23,398.11,501.19,630.96,794.33,1000.,1258.93,1584.89,1995.26,2511.89,3162.28,3981.07,5011.87,6309.57,7943.28,10000.]

settings = Dict()
settings["n_frequency_bands"] = 24
settings["tones_under_800Hz"] = false

# Compute
println("--- Compute ---")
@time f_pnlt_fwd!(y, x)
println(y)

# Compute partials
println("\n--- Compute partials ----")
Y = zeros(1)
X = [0., 0., 70., 62., 70., 80., 82., 83., 76., 80., 80., 79., 78., 80., 78., 76., 79., 85., 79., 78., 71., 60., 54., 45.]
J = Y.*X'
#'

const f_tape = JacobianTape(f_pnl!, Y, X)
const compiled_f_tape = compile(f_tape)

@time jacobian!(J, compiled_f_tape, x)
println(J)