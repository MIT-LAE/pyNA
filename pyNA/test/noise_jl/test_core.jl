using BenchmarkTools
using CSV
using DataFrames
using Test
using Interpolations

include("../../src/noise_src_jl/interpolation_functions.jl")
include("../../src/noise_src_jl/core.jl")

struct Settings_core
	r_0
	A_e
	N_f
	p_ref
	suppression
	case_name
end
settings = Settings_core(0.3048, 1., 24, 2e-5, true, "nasa_stca_standard")

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

struct Pyna_ip
	f_D_core
	f_S_core
end
f_D_core, f_S_core = get_core_interpolation_functions()
pyna_ip = Pyna_ip(f_D_core, f_S_core)


## Test 1
@testset "core module" begin
	n_t = 1
	M_0 = 0.3
	theta = 30.
	mdoti_c_star = 0.1
	Tti_c_star = 1.
	Ttj_c_star = 2.
	Pti_c_star = 3.
	DTt_des_c_star = 2.
	TS = 1.
	spl = zeros(24,)

	core_ge!(spl, pyna_ip, settings, ac, data.f, M_0, theta, TS, mdoti_c_star, Tti_c_star, Ttj_c_star, Pti_c_star, DTt_des_c_star)
	sol = [0.03137959298614686, 0.07294659038878948, 0.1764604190376361, 0.44324853227913186, 0.9989599424280967, 2.0074014176785835, 3.914120813807233, 7.063044094881777, 11.536606255882484, 16.921379339063808, 22.120424272303016, 20.427923214993164, 15.016955983695123, 9.838742602247473, 5.83479700959865, 3.1073018439188447, 1.5936151241511376, 0.7681916736824438, 0.3223816866189112, 0.1283424610981189, 0.05520935482569575, 0.022822875600450553, 0.008408684430149319, 0.0032238168661891217]

	@test(isapprox(spl, sol, rtol=1e-6))
end

## Test 2: timing
n_t = 1
M_0 = 0.3
theta = 30.
mdoti_c_star = 0.1
Tti_c_star = 1.
Ttj_c_star = 2.
Pti_c_star = 3.
DTt_des_c_star = 2.
TS = 1.
spl = zeros(24,)

@btime core_ge!(spl, pyna_ip, settings, ac, data.f, M_0, theta, TS, mdoti_c_star, Tti_c_star, Ttj_c_star, Pti_c_star, DTt_des_c_star)
