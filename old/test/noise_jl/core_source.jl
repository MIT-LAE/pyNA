using ReverseDiff: JacobianTape, jacobian!, compile
include("../../src/noise_src_jl/core_source.jl")
include("../../src/noise_src_jl/get_interpolation_functions.jl")

# Inputs 
struct PynaInterpolations
	f_D_core
	f_S_core
end
f_D_core, f_S_core = get_core_interpolation_functions()
pyna_ip = PynaInterpolations(f_D_core, f_S_core)

struct Aircraft
	n_eng
end
ac = Aircraft(3)

f = [50.11,63.09,79.43,100.,125.89,158.49,199.53,251.19,316.23,398.11,501.19,630.96,794.33,1000.,1258.93,1584.89,1995.26,2511.89,3162.28,3981.07,5011.87,6309.57,7943.28,10000.]

settings = Dict()
settings["r_0"] = 0.3048
settings["A_e"] = 10.669
settings["p_ref"] = 2e-5
settings["core_jet_suppression"] = false
settings["case_name"] = "stca"

M_0 = 0.3
theta = 30.
TS = 1.
mdoti_c_star = 0.1
Tti_c_star = 1
Ttj_c_star = 1.2
Pti_c_star = 10.
DTt_des_c_star = 10.
x = vcat(M_0, theta, TS, mdoti_c_star, Tti_c_star, Ttj_c_star, Pti_c_star, DTt_des_c_star)
y = zeros(24)

# Compute
println("--- Compute ---")
@time core_source_ge_fwd!(y, x)
println(y)

# Compute partials
println("\n--- Compute partials ----")
Y = zeros(24)
X = zeros(26)
J = Y.*X'
#'

const f_tape = JacobianTape(core_source_ge_fwd!, Y, X)
const compiled_f_tape = compile(f_tape)

@time jacobian!(J, compiled_f_tape, x)
println(J)




