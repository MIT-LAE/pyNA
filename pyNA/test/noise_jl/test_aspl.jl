using BenchmarkTools
using Test

include("../../src/noise_src_jl/interpolation_functions.jl")
include("../../src/noise_src_jl/aspl.jl")
include("../../src/noise_src_jl/oaspl.jl")

struct Data_aspl
	f
	aw_freq
	aw_db
end
l_i = 16
f = 10 .^ (0.1 * range(1+l_i, 40, length=24))
aw_freq = [10, 20, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
aw_db = [-70.4, -50.4, -34.5, -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8, 0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5, -0.1, -1.1, -2.5, -4.3, -6.7, -9.3]
data = Data_aspl(f, aw_freq, aw_db)
f_aw = get_a_weighting_interpolation_functions(data)


## Test 1: A-weighting with delta-dB weights (example NASA STCA Standard)
@testset "aspl module" begin
	spl = [58.28, 57.84, 56.18, 53.52, 49.69, 44.04, 37.77, 30.59, 27.62, 27.52, 25.01, 19.17, 8.14, -1.53, -6.88, -24.86, -34.92, -57.23, -82.45, -117.32, -154.43, -180.96, -180.96, -180.96]
	@test(isapprox(f_aspl(data.f, spl), 40.7063505, rtol=1e-6))
end

## Test 2: timing
spl = [58.28, 57.84, 56.18, 53.52, 49.69, 44.04, 37.77, 30.59, 27.62, 27.52, 25.01, 19.17, 8.14, -1.53, -6.88, -24.86, -34.92, -57.23, -82.45, -117.32, -154.43, -180.96, -180.96, -180.96]
@btime f_aspl(data.f, spl)
