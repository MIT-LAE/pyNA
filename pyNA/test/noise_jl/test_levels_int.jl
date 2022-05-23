using PCHIPInterpolation
using Interpolations

include("../../src/noise_src_jl/ioaspl.jl")
include("../../src/noise_src_jl/ipnlt.jl")
include("../../src/noise_src_jl/epnl.jl")

# Inputs
n_t = 100
t_o = range(1,100, length=100)
pnlt = 100 .- (range(1, 10, length=100).-9).^2

@time f_ipnlt(t_o, pnlt)
@time f_ipnlt(t_o, pnlt)

@time f_ioaspl(t_o, pnlt)
@time f_ioaspl(t_o, pnlt)

@time f_epnl(t_o, pnlt)
@time f_epnl(t_o, pnlt)
