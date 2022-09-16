using ReverseDiff: JacobianTape, jacobian!, compile
using CSV
using DataFrames
include("../../src/noise_src_jl/propagation.jl")
include("../../src/noise_src_jl/get_interpolation_functions.jl")


# Inputs 
struct Data
	abs_alt
	abs_freq
	abs
	Faddeeva_itau_im
	Faddeeva_itau_re
	Faddeeva_real
	Faddeeva_imag
end
df_abs = Matrix(DataFrame(CSV.File("../../data/isa/atmospheric_absorption.csv")))
df_re = Matrix(DataFrame(CSV.File("../../data/propagation/Faddeeva_real_small.csv")))
df_im = Matrix(DataFrame(CSV.File("../../data/propagation/Faddeeva_imag_small.csv")))
data = Data(df_abs[2:end, 1], df_abs[1,2:end], df_abs[2:end,2:end], df_re[2:end, 1], df_re[1, 2:end], df_re[2:end,2:end], df_im[2:end,2:end])

struct PynaInterpolations
	f_abs
	f_faddeeva_real
	f_faddeeva_imag
end

f_abs = get_atmospheric_absorption_interpolation_functions(data)
f_faddeeva_real, f_faddeeva_imag = get_ground_effects_interpolation_functions(data)
pyna_ip = PynaInterpolations(f_abs, f_faddeeva_real, f_faddeeva_imag)

settings = Dict()
settings["r_0"] = 0.3048
settings["n_frequency_bands"] = 24
settings["n_frequency_subbands"] = 5
settings["ground_resistance"] = 291.0 * 515.379
settings["incoherence_constant"] = 0.01
settings["direct_propagation"] = true
settings["absorption"] = true
settings["ground_effects"] = true

r = 2000.
z = 100.
c_bar = 300.
rho_0 = 1.2
I_0 = 420.
beta = 20.
x = vcat(r, z, c_bar, rho_0, I_0, beta)
y = ones(120)

f_sb = [   45.69436719,    47.8554422 ,    50.11872336,    52.48904442,   54.97146773,    57.52580003,    60.24643228,    63.09573445,   66.07979186,    69.20497765,    72.42069149,    75.84576457,   79.43282347,    83.18952918,    87.12390499,    91.17224886,   95.48416039,   100.        ,   104.72941228,   109.68249797,  114.77906094,   120.20743594,   125.89254118,   131.84651848,  138.08208392,   144.49827655,   151.33219579,   158.48931925,  165.98493258,   173.83504436,   181.91255231,   190.5159469 ,  199.5262315 ,   208.9626496 ,   218.84535481,   229.01433483,  239.84536691,   251.18864315,   263.06838969,   275.50997842,  288.31196578,   301.9474273 ,   316.22776602,   331.18348082,  346.84651304,   362.96326025,   380.12928926,   398.10717055,  416.93529997,   436.65388926,   456.94367188,   478.55442202,  501.18723363,   524.89044421,   549.71467735,   575.25800028,  602.4643228 ,   630.95734448,   660.79791862,   692.04977655,  724.20691489,   758.45764568,   794.32823472,   831.89529182,  871.23904992,   911.72248856,   954.84160391,  1000.        , 1047.29412282,  1096.82497969,  1147.79060935,  1202.0743594 , 1258.92541179,  1318.46518484,  1380.82083923,  1444.98276553, 1513.32195792,  1584.89319246,  1659.84932576,  1738.35044364, 1819.12552313,  1905.15946905,  1995.26231497,  2089.62649595, 2188.4535481 ,  2290.14334831,  2398.4536691 ,  2511.88643151, 2630.68389691,  2755.09978424,  2883.11965784,  3019.47427305, 3162.27766017,  3311.83480822,  3468.4651304 ,  3629.6326025 , 3801.2928926 ,  3981.07170553,  4169.35299973,  4366.53889259, 4569.43671876,  4785.54422016,  5011.87233627,  5248.90444211, 5497.14677346,  5752.58000284,  6024.64322803,  6309.5734448 , 6607.97918625,  6920.49776548,  7242.06914895,  7584.57645675, 7943.28234724,  8318.95291817,  8712.39049922,  9117.22488558, 9548.4160391 , 10000.        , 10472.94122821, 10968.24979695]

x_obs = [3500., 450., 1.2]

# Compute
println("--- Compute ---")
@time propagation_fwd!(y, x)
println(y)

# Compute partials
println("\n--- Compute partials ----")
Y = ones(120)
X = ones(6)
J = Y.*X'
#'

const f_tape = JacobianTape(propagation_fwd!, Y, X)
const compiled_f_tape = compile(f_tape)

@time jacobian!(J, compiled_f_tape, x)
println(J)




