using BenchmarkTools
using CSV
using DataFrames
using Test
using Interpolations

include("../../src/noise_src_jl/interpolation_functions.jl")
include("../../src/noise_src_jl/airframe.jl")

struct Settings_af
	r_0
	A_e
	N_f
	p_ref
	suppression
	case_name
	hsr_calibration
end
settings = Settings_af(0.3048, 1., 24, 2e-5, true, "nasa_stca_standard", true)

struct Data_af
	f
	supp_af_freq
	supp_af_angles
	supp_af
end
# Data
l_i = 16
f = 10 .^ (0.1 * range(1+l_i, 40, length=24))

data_hsr = DataFrame(CSV.File("../../data/sources/airframe/hsr_suppression.csv"))
supp_af_freq = Array(data_hsr)[2:end, 1]
supp_af_angles = Array(data_hsr)[1, 2:end]
supp_af = Array(data_hsr)[2:end, 2:end]
data = Data_af(f, supp_af_freq, supp_af_angles, supp_af)

struct Ac_af
	n_eng
	comp_lst
	af_S_h
	af_S_v
	af_S_w
	af_b_f
	af_b_h
	af_b_v
	af_b_w
	af_S_f
	af_s
	af_d_mg
	af_d_ng
	af_l_mg
	af_l_ng
	af_n_mg
	af_n_ng
	af_N_mg
	af_N_ng
	c_d_g
	mu_r
	B_fan
	V_fan
	RSS_fan
	M_d_fan
	inc_F_n
	TS_lower
	TS_upper
	af_clean_w
	af_clean_h
	af_clean_v
	af_delta_wing
end
ac = Ac_af(3, ["wing", "tail_h", "les", "tef", "lg"], 20.16, 21.3677, 150.41, 6.096, 5.6388, 4.7244, 20.51304, 11.1484, 1.0, 0.9144, 0.82296, 2.286, 1.8288, 4.0,   2.0,   2.0,   1.0, 0.0240,   0.0175, 25,   48,   300.0, 1.68, 0.25,  0.65,   1.0,   true, false,  true,   true)

struct Pyna_ip
	f_hsr_supp
end
f_hsr_supp = get_airframe_interpolation_functions(data)
pyna_ip = Pyna_ip(f_hsr_supp)


@testset "airframe module" begin
	## Test 1
	M_0 = 0.3
	mu_0 = 1.78e-5
	c_0 = 340.
	rho_0 = 1.
	theta = 30.
	phi = 10.
	theta_flaps = 10.
	I_landing_gear = 1
	spl = zeros(24,)

	airframe!(spl, pyna_ip, settings, ac, data.f, M_0, mu_0, c_0, rho_0, theta, phi, theta_flaps, I_landing_gear)
	sol = [3.9091806708511148, 8.876480031357033, 18.620896223638503, 36.75071365286709, 66.09140042606035, 79.53377673535167, 64.81144402366124, 59.25972104103154, 72.68725867350331, 74.84474277157794, 74.95203444555902, 52.471301173694734, 47.97525644457571, 41.42944151424544, 46.80701403454354, 36.52302400759434, 27.31111047407081, 19.966746449542566, 14.392394807145768, 10.303814963929455, 7.369831588261831, 5.289680499669068, 3.8216076326366455, 3.8435911046483966]

	@test(isapprox(spl, sol, rtol=1e-6))
end

## Test 2: timing
M_0 = 0.3
mu_0 = 1.78e-5
c_0 = 340.
rho_0 = 1.
theta = 30.
phi = 10.
theta_flaps = 10.
I_landing_gear = 1
spl = zeros(24,)

@btime airframe!(spl, pyna_ip, settings, ac, data.f, M_0, mu_0, c_0, rho_0, theta, phi, theta_flaps, I_landing_gear)


