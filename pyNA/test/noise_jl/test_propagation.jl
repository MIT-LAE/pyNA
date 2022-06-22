using BenchmarkTools
using Test
using CSV
using DataFrames

include("../../src/noise_src_jl/interpolation_functions.jl")
include("../../src/noise_src_jl/direct_propagation.jl")
include("../../src/noise_src_jl/atmospheric_absorption.jl")
include("../../src/noise_src_jl/ground_reflections.jl")
include("../../src/noise_src_jl/split_subbands.jl")
include("../../src/noise_src_jl/propagation.jl")

struct Settings_prop
	N_f
	N_b
	sigma
	a_coh
	r_0
	direct_propagation
	absorption
	groundeffects
	lateral_attenuation
end
settings = Settings_prop(24, 5, 291.0 * 515.379, 0.01, 0.3048, true, true, true, false)

struct Data_prop
	f_sb
	abs_alt
	abs_freq
	abs
	Faddeeva_itau_re 
	Faddeeva_itau_im
	Faddeeva_real
	Faddeeva_imag
end

l_i = 16
f = 10 .^ (0.1 * range(1+l_i, 40, length=24))
f_sb = zeros(settings.N_b * settings.N_f)
m = (settings.N_b - 1) / 2.
w = 2. ^ (1 / (3. * settings.N_b))
for k in 0:1:settings.N_f-1
    for h in 0:1:settings.N_b-1
        f_sb[k * settings.N_b + h+1] = w ^ (h - m) * f[k+1]
    end
end
data_absorption = DataFrame(CSV.File("../../data/isa/atmospheric_absorption.csv"))
abs_alt = Array(data_absorption)[2:end, 1]
abs_freq = Array(data_absorption)[1, 2:end]
abs = Array(data_absorption)[2:end, 2:end]
data_fadd_real = DataFrame(CSV.File("../../data/propagation/Faddeeva_real_small.csv"))
data_fadd_imag = DataFrame(CSV.File("../../data/propagation/Faddeeva_imag_small.csv"))
Faddeeva_itau_re = Array(data_fadd_real)[1, 2:end]
Faddeeva_itau_im = Array(data_fadd_imag)[2:end, 1]
Faddeeva_real = Array(data_fadd_real)[2:end, 2:end]
Faddeeva_imag = Array(data_fadd_imag)[2:end, 2:end]
data = Data_prop(f_sb, abs_alt, abs_freq, abs, Faddeeva_itau_re,Faddeeva_itau_im,Faddeeva_real,Faddeeva_imag)

struct Pyna_ip
	f_abs
	f_faddeeva_real
	f_faddeeva_imag
end	
f_abs, f_faddeeva_real, f_faddeeva_imag = get_propagation_interpolation_functions(data)
pyna_ip = Pyna_ip(f_abs, f_faddeeva_real, f_faddeeva_imag)

## Test 1: example from the nasa stca
@testset "propagation module" begin
    
	x_obs = [6499.860000000001, 0.0, 1.2192]
	r = 895.7785893881709
	x = 1000.
	z = 1000.
	beta = 49.93907595256916
	c_bar = 349.9860553285613
	rho_0 = 1.11302898
	I_0 = 410.
	beta = 30.
	spl = ones(24,)

	propagation!(spl, pyna_ip, settings, data.f_sb, x_obs, r, x, z, c_bar, rho_0, I_0, beta)
	sol = [2.8087653303005795e-7, 2.1587239184906178e-7, 1.380546559251876e-7, 5.95306858699159e-8, 8.865161131463598e-9, 2.866447634891456e-8, 1.3875875251217116e-7, 2.537297450591298e-7, 1.9100184179644908e-7, 2.931080280397527e-8, 1.1982049593714838e-7, 7.412861789212705e-8, 7.448635496987071e-8, 3.804019240306856e-8, 3.326670509075246e-8, 2.1513402489436735e-8, 1.1227168486017006e-8, 6.293280699740046e-9, 2.570546288453019e-9, 9.07901136926327e-10, 3.726295326629245e-10, 7.600218421249636e-11, 6.399130983872633e-12, 1.5907895725751865e-13]
	@test(isapprox(spl, sol, rtol=1e-6))

end

## Test 2: timing
x_obs = [6499.860000000001, 0.0, 1.2192]
r = 895.7785893881709
x = 1000.
z = 1000.
beta = 49.93907595256916
c_bar = 349.9860553285613
rho_0 = 1.11302898
I_0 = 410.
beta = 30.
spl = ones(24,)

@btime propagation!(spl, pyna_ip, settings, data.f_sb, x_obs, r, x, z, c_bar, rho_0, I_0, beta)

