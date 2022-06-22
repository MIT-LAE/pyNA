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
f_noy = get_noy_interpolation_functions(data)


@testset "tone corrections module" begin
    
	## Test 1: Example of tone correction calculation for a turbofan engine (ICAO Annex 16 p. App 1-14) - F (step 10)
	spl = [0., 0., 70., 62., 70., 80., 82., 83., 76., 80., 80., 79., 78., 80., 78., 76., 79., 85., 79., 78., 71., 60., 54., 45.]
	sol = f_tone_corrections(settings, spl)
	C = [0.0, 0.0, 0.0, 0.0, 0.0, 0.27777777777777, 0.05555555555555, 0.6666666666666666, 0.0, 0.1666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.33333333333333, 0.0, 0.0, 0.0, 0.0]
	c_max = 2

	@test(isapprox(sol, c_max, rtol=1e-6))

end

## Test 2: timing
spl = [0., 0., 70., 62., 70., 80., 82., 83., 76., 80., 80., 79., 78., 80., 78., 76., 79., 85., 79., 78., 71., 60., 54., 45.]
@btime f_tone_corrections(settings, spl)