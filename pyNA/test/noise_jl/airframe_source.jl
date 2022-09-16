using ReverseDiff: JacobianTape, jacobian!, compile
using CSV
using DataFrames
include("../../src/noise_src_jl/airframe_source.jl")
include("../../src/noise_src_jl/get_interpolation_functions.jl")

# Inputs 
struct Data
	supp_af_freq
	supp_af_angles
	supp_af
end
df = Matrix(DataFrame(CSV.File("../../data/sources/airframe/hsr_suppression.csv")))
data = Data(df[2:end, 1], df[1,2:end], df[2:end,2:end])

struct PynaInterpolations
	f_hsr_supp
end
f_hsr_supp = get_airframe_interpolation_functions(data)
pyna_ip = PynaInterpolations(f_hsr_supp)

struct Aircraft
    mtow
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
af = Aircraft(55000.0,3,["wing", "tail_h", "les", "tef", "lg"],   20.16,21.3677,150.41, 6.096,5.6388,4.7244,20.51304,11.1484,1.0,0.9144,0.82296,2.286,1.8288,4.0,  2.0,  2.0,  1.0,0.0240,  0.0175,25,  48,  300.0,1.68,0.25, 0.65,  1.0,  true,false, true,  true)

f = [50.11,63.09,79.43,100.,125.89,158.49,199.53,251.19,316.23,398.11,501.19,630.96,794.33,1000.,1258.93,1584.89,1995.26,2511.89,3162.28,3981.07,5011.87,6309.57,7943.28,10000.]

settings = Dict()
settings["r_0"] = 0.3048
settings["A_e"] = 10.669
settings["p_ref"] = 2e-5
settings["core_jet_suppression"] = false
settings["case_name"] = "stca"
settings["n_frequency_bands"] = 24

M_0 = 0.3
c_0 = 300.
rho_0 = 1.2
mu_0 = 1e-5
theta = 30.
phi = 10.
theta_flaps = 10.
I_landing_gear = 1
x = vcat(M_0, c_0, rho_0, mu_0, theta, phi, theta_flaps, I_landing_gear)
y = zeros(24)

# Compute
println("--- Compute ---")
@time airframe_source_fwd!(y, x)
println(y)

# Compute partials
println("\n--- Compute partials ----")
Y = zeros(24)
X = zeros(8)
J = Y.*X'
#'

const f_tape = JacobianTape(airframe_source_fwd!, Y, X)
const compiled_f_tape = compile(f_tape)

@time jacobian!(J, compiled_f_tape, x)
println(J)




