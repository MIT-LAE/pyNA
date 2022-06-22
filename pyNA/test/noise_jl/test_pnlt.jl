using BenchmarkTools
using CSV
using DataFrames
using Test

include("../../src/noise_src_jl/interpolation_functions.jl")
include("../../src/noise_src_jl/pnlt.jl")
data_noy_file = DataFrame(CSV.File("../../data/levels/spl_noy.csv"))

struct Settings_pnlt
	N_f
	TCF800
end
settings = Settings_pnlt(24, true)

struct Data_pnlt
	f
	noy_freq
	noy_spl
	noy
end
l_i = 16
f = 10 .^ (0.1 * range(1+l_i, 40, length=24))
noy_freq = Array(data_noy_file)[1, 2:end]
noy_spl = Array(data_noy_file)[2:end, 1]
noy = Array(data_noy_file)[2:end, 2:end]
data = Data_pnlt(f, noy_freq, noy_spl, noy)

struct Pyna_ip
	f_noy
end
f_noy = get_noy_interpolation_functions(data)
pyna_ip = Pyna_ip(f_noy)

@testset "pnlt module" begin
    
	## Test 1: Example of pnlt calculation using nasa standard package
	spl = [58.28, 57.85, 56.18, 53.53, 49.70, 44.08, 37.89, 30.84, 28.08, 28.14, 25.76, 20.11, 9.24, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99]
	sol = f_pnlt(pyna_ip, settings, data.f, spl)
	@test(isapprox(sol[1], 49.35, rtol=1e-2))

end

## Test 2: timing
spl = [58.28, 57.85, 56.18, 53.53, 49.70, 44.08, 37.89, 30.84, 28.08, 28.14, 25.76, 20.11, 9.24, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99]
@btime f_pnlt(pyna_ip, settings, data.f, spl)

