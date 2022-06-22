using BenchmarkTools
using CSV
using DataFrames
using Test
using Interpolations
using NPZ

include("../../src/noise_src_jl/interpolation_functions.jl")
include("../../src/noise_src_jl/jet.jl")

struct Settings_jet
	r_0
	A_e
	N_f
	p_ref
	suppression
	case_name
end
settings = Settings_jet(0.3048, 1., 24, 2e-5, true, "nasa_stca_standard")

struct Data_jet
	f
	jet_D_velocity
	jet_D_angles
	jet_D
	jet_xi_velocity
	jet_xi_angles
	jet_xi
	jet_F_angles
	jet_F_temperature
	jet_F_velocity
	jet_F_strouhal
	jet_F
end


# Data
l_i = 16
f = 10 .^ (0.1 * range(1+l_i, 40, length=24))
data_D_jet = DataFrame(CSV.File("../../data/sources/jet/directivity_function.csv"))
data_xi_jet = DataFrame(CSV.File("../../data/sources/jet/strouhal_correction.csv"))
jet_D_velocity = Array(data_D_jet)[2:end,1]
jet_D_angles = Array(data_D_jet)[1,2:end]
jet_D = Array(data_D_jet)[2:end,2:end]
jet_xi_velocity = Array(data_xi_jet)[2:end, 1]
jet_xi_angles = Array(data_xi_jet)[1, 2:end]
jet_xi = Array(data_xi_jet)[2:end, 2:end]
jet_F_angles = [0., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180.]
jet_F_temperature = [0., 1., 2., 2.5, 3., 3.5, 4., 5., 6., 7.]
jet_F_velocity = [-0.4, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.4]
jet_F_strouhal = [-2., -1.6, -1.3, -1.15, -1, -0.824, -0.699, -0.602, -0.5, -0.398, -0.301, -0.222, 0., 0.477, 1., 1.6, 1.7, 2.5]
jet_F = npzread("../../data/sources/jet/spectral_function_extended_T.npy")
data = Data_jet(f, jet_D_velocity,jet_D_angles, jet_D, jet_xi_velocity, jet_xi_angles, jet_xi, jet_F_angles, jet_F_temperature, jet_F_velocity, jet_F_strouhal, jet_F)

struct Ac_jet
	n_eng
end
ac = Ac_jet(3)

struct Pyna_ip
	f_omega_jet
	f_log10P_jet
	f_log10D_jet
	f_xi_jet
	f_log10F_jet
	f_m_theta_jet
end
f_omega_jet, f_log10P_jet, f_log10D_jet, f_xi_jet, f_log10F_jet, f_m_theta_jet = get_jet_mixing_interpolation_functions(data)
pyna_ip = Pyna_ip(f_omega_jet, f_log10P_jet, f_log10D_jet, f_xi_jet, f_log10F_jet, f_m_theta_jet)


## Test 1
@testset "jet mixing module" begin
	M_0 = 0.3
	c_0 = 340.
	theta = 30.
	TS = 1.
	V_j_star = 2.
	rho_j_star = 2.
	A_j_star = 1.
	Tt_j_star = 2.
	spl = zeros(24,)

	jet_mixing!(spl, pyna_ip, settings, ac, data.f, M_0, c_0, theta, TS, V_j_star, rho_j_star, A_j_star, Tt_j_star)
	sol = [560669.748128467, 784057.5270300661, 1.0881213311100388e6, 1.515923634865849e6, 2.0254123363904704e6, 2.445279434684468e6, 2.755080003881708e6, 2.9726082507890435e6, 2.9750958977145157e6, 2.85419046592108e6, 2.7381985306802713e6, 2.371521555128568e6, 2.0320815020384304e6, 1.7412261010222605e6, 1.492001345339663e6, 1.26563506643017e6, 1.0066583793114823e6, 800674.001152869, 636838.5435390892, 506527.4130957526, 402880.7973707804, 303807.92599611386, 227823.80445378905, 170843.75170798632]

	@test(isapprox(spl, sol, rtol=1e-6))
end

## Test 2: timing
M_0 = 0.3
c_0 = 340.
theta = 30.
TS = 1.
V_j_star = 2.
rho_j_star = 2.
A_j_star = 1.
Tt_j_star = 2.
spl = zeros(24,)

@btime jet_mixing!(spl, pyna_ip, settings, ac, data.f, M_0, c_0, theta, TS, V_j_star, rho_j_star, A_j_star, Tt_j_star)	