using BenchmarkTools
using CSV
using DataFrames
using Test
using Interpolations
using NPZ

include("../../src/noise_src_jl/interpolation_functions.jl")
include("../../src/noise_src_jl/fan.jl")

struct Settings_fan
	r_0
	A_e
	N_f
	p_ref
	suppression
	case_name
	fan_BB_method
	fan_RS_method
	combination_tones
	fan_liner_suppression
	fan_id
	fan_igv
	n_harmonics
	shielding
end
settings = Settings_fan(0.3048, 1., 24, 2e-5, true, "nasa_stca_standard", "geae", "allied_signal", false, true, false, false, 10, true)

struct Data_fan
	f
	supp_fi_angles
    supp_fi_freq
    supp_fi
    supp_fd_angles
    supp_fd_freq
    supp_fd
end


# Data
l_i = 16
f = 10 .^ (0.1 * range(1+l_i, 40, length=24))
data_fi_suppression = DataFrame(CSV.File("../../data/sources/fan/liner_inlet_suppression.csv"))
data_fd_suppression = DataFrame(CSV.File("../../data/sources/fan/liner_discharge_suppression.csv"))
supp_fi_angles = Array(data_fi_suppression)[1, 2:end]
supp_fi_freq = Array(data_fi_suppression)[2:end,1]
supp_fi = Array(data_fi_suppression)[2:end, 2:end]
supp_fd_angles = Array(data_fd_suppression)[1, 2:end]
supp_fd_freq = Array(data_fd_suppression)[2:end,1]
supp_fd = Array(data_fd_suppression)[2:end, 2:end]
data = Data_fan(f, supp_fi_angles, supp_fi_freq, supp_fi, supp_fd_angles, supp_fd_freq, supp_fd)

struct Ac_fan
	n_eng
	B_fan
	V_fan
	RSS_fan
	M_d_fan
end
ac = Ac_fan(3, 25, 48, 300.0, 1.68)

struct Pyna_ip
	f_supp_fi
	f_supp_fd
	f_F3IB
	f_F3DB
	f_F3TI
	f_F3TD
	f_F2CT
	f_TCS_takeoff_ih1
	f_TCS_takeoff_ih2
	f_TCS_approach_ih1
	f_TCS_approach_ih2
end
f_supp_fi, f_supp_fd, f_F3IB, f_F3DB, f_F3TI, f_F3TD, f_F2CT, f_TCS_takeoff_ih1, f_TCS_takeoff_ih2, f_TCS_approach_ih1, f_TCS_approach_ih2 = get_fan_interpolation_functions(settings, data)
pyna_ip = Pyna_ip(f_supp_fi, f_supp_fd, f_F3IB, f_F3DB, f_F3TI, f_F3TD, f_F2CT, f_TCS_takeoff_ih1, f_TCS_takeoff_ih2, f_TCS_approach_ih1, f_TCS_approach_ih2)



@testset "fan module" begin
	## Test 1
	shield = zeros(24,)
	M_0 = 0.4
	c_0 = 340.
	T_0 = 300.
	rho_0 = 1.
	theta = 30.
	DTt_f_star = 0.5
	mdot_f_star = 0.2
	N_f_star = 0.4
	A_f_star = 1.
	d_f_star = 1.
	comp = "fan_inlet"
	spl = zeros(24,)
	fan!(spl, pyna_ip, settings, ac, data.f, shield, M_0, c_0, T_0, rho_0, theta, DTt_f_star, mdot_f_star, N_f_star, A_f_star, d_f_star, comp)
	sol = [4.863346104568135e-7, 3.5165832768044766e-6, 2.3151861610288564e-5, 0.00013862785090516455, 0.0007546809207254557, 0.0037326575090817085, 0.016770204109453495, 0.06854341905329474, 0.25514405592760636, 0.8675253227299412, 2.7035578014379302, 7.764777696204669, 20.688934356658883, 51.382074248138984, 120.09574287850953, 265.3371733503996, 556.3285291799235, 1110.9552502733052, 2108.2786851167793, 3784.7761107450074, 61177.24474798411, 9978.377185188847, 14357.890022102298, 28736.324298172665]
	@test(isapprox(spl, sol, rtol=1e-6))
end

## Test 2: timing
shield = zeros(24,)
M_0 = 0.4
c_0 = 340.
T_0 = 300.
rho_0 = 1.
theta = 30.
DTt_f_star = 0.5
mdot_f_star = 0.2
N_f_star = 0.4
A_f_star = 1.
d_f_star = 1.
comp = "fan_inlet"
spl = zeros(24,)
@btime fan!(spl, pyna_ip, settings, ac, data.f, shield, M_0, c_0, T_0, rho_0, theta, DTt_f_star, mdot_f_star, N_f_star, A_f_star, d_f_star, comp)

