using ReverseDiff: JacobianTape, jacobian!, compile
using CSV
using DataFrames
include("../../src/noise_src_jl/fan_source.jl")
include("../../src/noise_src_jl/get_interpolation_functions.jl")

# Inputs 
struct Data
	supp_fi_angles
	supp_fi_freq
	supp_fi
	supp_fd_angles
	supp_fd_freq
	supp_fd
end
df_i = Matrix(DataFrame(CSV.File("../../data/sources/fan/liner_inlet_suppression.csv")))
df_d = Matrix(DataFrame(CSV.File("../../data/sources/fan/liner_discharge_suppression.csv")))
data = Data(df_i[1, 2:end], df_i[2:end, 1], df_i[2:end, 2:end], df_d[1, 2:end], df_d[2:end, 1], df_d[2:end, 2:end])

struct PynaInterpolations
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

settings = Dict()
settings["r_0"] = 0.3048
settings["A_e"] = 10.669
settings["p_ref"] = 2e-5
settings["core_jet_suppression"] = false
settings["case_name"] = "stca"
settings["fan_BB_method"] = "geae"
settings["fan_RS_method"] = "allied_signal"
settings["fan_id"] = false
settings["fan_igv"] = false
settings["n_frequency_bands"] = 24
settings["n_harmonics"] = 10
settings["fan_ge_flight_cleanup"] = "takeoff"
settings["fan_combination_tones"] = false
settings["fan_liner_suppression"] = false
settings["shielding"] = true

shield = ones(24)

f_supp_fi, f_supp_fd, f_F3IB, f_F3DB, f_F3TI, f_F3TD, f_F2CT, f_TCS_takeoff_ih1, f_TCS_takeoff_ih2, f_TCS_approach_ih1, f_TCS_approach_ih2 = get_fan_interpolation_functions(settings, data)
pyna_ip = PynaInterpolations(f_supp_fi, f_supp_fd, f_F3IB, f_F3DB, f_F3TI, f_F3TD, f_F2CT, f_TCS_takeoff_ih1, f_TCS_takeoff_ih2, f_TCS_approach_ih1, f_TCS_approach_ih2)

struct Aircraft
	n_eng
	B_fan
	V_fan
	M_d_fan
	RSS_fan
end
ac = Aircraft(3, 25, 48, 1.68, 300.)

f = [50.11,63.09,79.43,100.,125.89,158.49,199.53,251.19,316.23,398.11,501.19,630.96,794.33,1000.,1258.93,1584.89,1995.26,2511.89,3162.28,3981.07,5011.87,6309.57,7943.28,10000.]

comp = "fan_inlet"
M_0 = 0.3
c_0 = 300.
T_0 = 290.
rho_0 = 1.2
theta = 50.
DTt_f_star = 0.3
mdot_f_star = 0.3
N_f_star = 0.6
A_f_star = 1.
d_f_star = 1.
x = vcat(M_0, c_0, T_0, rho_0, theta, DTt_f_star, mdot_f_star, N_f_star, A_f_star, d_f_star)
y = zeros(24)

# Compute
println("--- Compute ---")
@time fan_source_fwd!(y, x)
println(y)

# Compute partials
println("\n--- Compute partials ----")
Y = zeros(24)
X = vcat(M_0, c_0, T_0, rho_0, theta, DTt_f_star, mdot_f_star, N_f_star, A_f_star, d_f_star)
J = Y.*X'
#'

const f_tape = JacobianTape(fan_source_fwd!, Y, X)
const compiled_f_tape = compile(f_tape)

@time jacobian!(J, compiled_f_tape, x)
println(J)




