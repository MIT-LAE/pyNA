using BenchmarkTools
using CSV
using DataFrames
using Test
using Interpolations

include("../../src/noise_src_jl/interpolation.jl")
include("../../src/noise_src_jl/core.jl")
include("../../src/noise_src_jl/core2.jl")

struct Settings_core
	r_0
	A_e
	N_f
	p_ref
end
settings = Settings_core(0.3048, 1., 24, 2e-5)

struct Data_core
	f
end
# Data
l_i = 16
f = 10 .^ (0.1 * range(1+l_i, 40, length=24))
data = Data_core(f)

struct Ac_core
	n_eng
end
ac = Ac_core(3)

# Test 1
n_t = 1
M_0 = [0.3]
theta = [30.]
mdoti_c_star = [0.1]
Tti_c_star = [1.]
Ttj_c_star = [2.]
Pti_c_star = [3.]
DTt_des_c_star = [2.]

@btime sol = core_ge(settings, data, ac, n_t, M_0, theta, mdoti_c_star, Tti_c_star, Ttj_c_star, Pti_c_star, DTt_des_c_star)
println(sol)

# Test 2
n_t = 1
M_0 = 0.3
theta = 30.
mdoti_c_star = 0.1
Tti_c_star = 1.
Ttj_c_star = 2.
Pti_c_star = 3.
DTt_des_c_star = 2.
spl = zeros(1, 24)

f_D_core, f_S_core = get_interpolation_functions_core()
@btime core_ge!(spl, settings, data, ac, f_D_core, f_S_core, M_0, theta, mdoti_c_star, Tti_c_star, Ttj_c_star, Pti_c_star, DTt_des_c_star)
println(spl)

