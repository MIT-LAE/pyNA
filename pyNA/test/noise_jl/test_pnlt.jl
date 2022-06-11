using BenchmarkTools
using CSV
using DataFrames
using Test
using Interpolations

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
# Data
l_i = 16
f = 10 .^ (0.1 * range(1+l_i, 40, length=24))
noy_freq = Array(data_noy_file)[1, 2:end]
noy_spl = Array(data_noy_file)[2:end, 1]
noy = Array(data_noy_file)[2:end, 2:end]
data = Data_pnlt(f, noy_freq, noy_spl, noy)

@testset "pnlt module" begin
    
	## Test 1: Example of tone correction calculation for a turbofan engine (ICAO Annex 16 p. App 1-14) - spl_pp (step 9)
	## Note: change the output of the function to spl_pp
    if false
    	# Inputs
		spl = reshape([0., 0., 70., 62., 70., 80., 82., 83., 76., 80., 80., 79., 78., 80., 78., 76., 79., 85., 79., 78., 71., 60., 54., 45.], (1, 24))

		sol = f_tone_corrections(settings, spl)

		spl_pp = reshape([0., 0., 70., 67.66666666666667, 71., 77.66666666666667, 80.3333333333333, 79., 77.66666666666667, 78., 79., 79., 79., 78.66666666666667, 78., 77.66666666666667, 78., 79., 78.66666666666667, 76., 69.66666666666667, 61.66666666666667, 53., 45.], (1, 24))

		@test(isapprox(sol, spl_pp, rtol=1e-6))	 
	end

    ## Test 2: Example of tone correction calculation for a turbofan engine (ICAO Annex 16 p. App 1-14) - F (step 10)
    ## Note: change the output of the function to F
    ## Note: the F function is noted down if F >= 1.5 (new certification rules after 6 october 1977)
	if false
	    # Inputs
		spl = reshape([0., 0., 70., 62., 70., 80., 82., 83., 76., 80., 80., 79., 78., 80., 78., 76., 79., 85., 79., 78., 71., 60., 54., 45.], (1, 24))

		sol = f_tone_corrections(settings, spl)

		F = reshape([0., 0., 0., 0., 0., 2.3333333333333, 1.66666666666667, 4., 0., 2., 0., 0., 0., 0., 0., 0., 0., 6., 0., 2., 0., 0., 0., 0.], (1, 24))

		@test(isapprox(sol, F, rtol=1e-6))	 		

	end

	## Test 3: Example of tone correction calculation for a turbofan engine (ICAO Annex 16 p. App 1-14) - F (step 10)
	if true
    	# Inputs
		spl = reshape([0., 0., 70., 62., 70., 80., 82., 83., 76., 80., 80., 79., 78., 80., 78., 76., 79., 85., 79., 78., 71., 60., 54., 45.], (1, 24))

		sol = f_tone_corrections(settings, spl)

		C = reshape([0.0, 0.0, 0.0, 0.0, 0.0, 0.27777777777777, 0.05555555555555, 0.6666666666666666, 0.0, 0.1666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.33333333333333, 0.0, 0.0, 0.0, 0.0], (1, 24))

		@test(isapprox(sol, C, rtol=1e-6))

	end	

	# Test 4: Example of pnlt calculation using nasa standard package
	if true
		# Inputs
		spl = reshape([58.28, 57.85, 56.18, 53.53, 49.70, 44.08, 37.89, 30.84, 28.08, 28.14, 25.76, 20.11, 9.24, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99, 1e-99], (1, 24))

		sol = f_pnlt(settings, data, spl)

		@test(isapprox(sol[1], [49.35], rtol=1e-2))

	end

end

