using BenchmarkTools
using CSV
using DataFrames
using Test
using Interpolations

include("../../src/noise_src_jl/interpolation.jl")
include("../../src/noise_src_jl/jet.jl")
include("../../src/noise_src_jl/jet2.jl")

struct Settings_jet_shock
	r_0
	A_e
	N_f
	p_ref
	N_shock
	suppression
	case_name
end
settings = Settings_jet_shock(0.3048, 1., 24, 2e-5, 8, true, "stca")

struct Data_jet_shock
	f
end
# Data
l_i = 16
f = 10 .^ (0.1 * range(1+l_i, 40, length=24))
data = Data_jet_shock(f)

struct Ac_jet_shock
	n_eng
end
ac = Ac_jet_shock(3)

# Test 1
M_0 = [0.3]
c_0 = [340.]
theta = [30.]
V_j_star = [1.5]
M_j = [1.1]
A_j_star = [1.1]
Tt_j_star = [3.]

f_C, f_H = get_interpolation_functions_jet_shock()
sol = jet_shock(settings, data, ac, 1, M_0, c_0, theta, V_j_star, M_j, A_j_star, Tt_j_star)

# Test 2
spl = zeros(1, 24)

M_0 = 0.3
c_0 = 340.
theta = 30.
V_j_star = 1.5
M_j = 1.1
A_j_star = 1.1
Tt_j_star = 3.

f_C, f_H = get_interpolation_functions_jet_shock()
jet_shock!(spl, settings, data, ac, M_0, c_0, theta, V_j_star, M_j, A_j_star, Tt_j_star)

