using BenchmarkTools
include("../../src/noise_src_jl/geometry.jl")

# Inputs
x = collect(range(0, 1000, length=100))
y = zeros(100)
z = collect(range(0, 100, length=100))
alpha = 10*ones(100)
gamma = 10*ones(100)
t_s = range(0, 50, length=100)
c_0 = 300*ones(100)
T_0 = 300*ones(100)
x_obs = [6500., 0., 1.2]
n_t = 100

# Settings struct
struct Settings
    dT::Float64
end
settings = Settings(0.)

# Compute geometry
@btime geometry(settings, x_obs, x, y, z, alpha, gamma, t_s, c_0, T_0)
