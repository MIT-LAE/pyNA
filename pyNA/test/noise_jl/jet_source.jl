using ReverseDiff: JacobianTape, jacobian!, compile
using CSV
using NPZ: npzread
using DataFrames
include("../../src/noise_src_jl/jet_source.jl")
include("../../src/noise_src_jl/get_interpolation_functions.jl")

# Inputs 
struct Data
	jet_D_angles
	jet_D_velocity
	jet_D
	jet_xi_angles
	jet_xi_velocity
	jet_xi
	jet_F_angles
	jet_F_temperature
	jet_F_velocity
	jet_F_strouhal
	jet_F
end
df_D = Matrix(DataFrame(CSV.File("../../data/sources/jet/directivity_function.csv")))
df_xi = Matrix(DataFrame(CSV.File("../../data/sources/jet/strouhal_correction.csv")))
df_F = npzread("../../data/sources/jet/spectral_function_extended_T.npy")


jet_a = [0, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
jet_t = [0, 1, 2, 2.5, 3, 3.5, 4, 5, 6, 7]
jet_v = [-0.4, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.4]
jet_s = [-2, -1.6, -1.3, -1.15, -1, -0.824, -0.699, -0.602, -0.5, -0.398, -0.301, -0.222, 0, 0.477, 1, 1.6, 1.7, 2.5]
data = Data(df_D[1, 2:end], df_D[2:end, 1], df_D[2:end, 2:end], 
			df_xi[1, 2:end], df_xi[2:end, 1], df_xi[2:end, 2:end],
			jet_a, jet_t, jet_v, jet_s, df_F)

struct PynaInterpolations
	f_omega_jet
	f_log10P_jet
	f_log10D_jet
	f_xi_jet
	f_log10F_jet
	f_m_theta_jet
end

f_omega_jet, f_log10P_jet, f_log10D_jet, f_xi_jet, f_log10F_jet, f_m_theta_jet = get_jet_mixing_interpolation_functions(data)
pyna_ip = PynaInterpolations(f_omega_jet, f_log10P_jet, f_log10D_jet, f_xi_jet, f_log10F_jet, f_m_theta_jet)

settings = Dict()
settings["r_0"] = 0.3048
settings["A_e"] = 10.669
settings["p_ref"] = 2e-5
settings["core_jet_suppression"] = false
settings["n_frequency_bands"] = 24
settings["case_name"] = "stca"
settings["n_shock"] = 8

struct Aircraft
	n_eng
end
af = Aircraft(3)

f = [50.11,63.09,79.43,100.,125.89,158.49,199.53,251.19,316.23,398.11,501.19,630.96,794.33,1000.,1258.93,1584.89,1995.26,2511.89,3162.28,3981.07,5011.87,6309.57,7943.28,10000.]

M_0 = 0.3
c_0 = 300.
theta = 50.
TS = 1.
V_j_star = 0.6
rho_j_star = 0.3
A_j_star = 0.2
Tt_j_star = 1.
x = vcat(M_0, c_0, theta, TS, V_j_star, rho_j_star, A_j_star, Tt_j_star)
y = ones(24)

# Compute
println("--- Compute ---")
@time jet_mixing_source_fwd!(y, x)
println(y)

# Compute partials
println("\n--- Compute partials ----")
Y = ones(24)
X = vcat(M_0, c_0, theta, TS, V_j_star, rho_j_star, A_j_star, Tt_j_star)
J = Y.*X'
#'

const f_tape = JacobianTape(jet_mixing_source_fwd!, Y, X)
const compiled_f_tape = compile(f_tape)

@time jacobian!(J, compiled_f_tape, x)
println(J)




