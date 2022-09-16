using ReverseDiff: JacobianTape, jacobian!, compile
include("../../src/noise_src_jl/tone_corrections.jl")

# Inputs 
x = [0., 0., 70., 62., 70., 80., 82., 83., 76., 80., 80., 79., 78., 80., 78., 76., 79., 85., 79., 78., 71., 60., 54., 45.]
y = zeros(24)

settings = Dict()
settings["n_frequency_bands"] = 24
settings["tones_under_800Hz"] = false

# Compute
println("--- Compute ---")
@time f_tone_corrections_fwd!(y, x)
println(y)

# Compute partials
println("\n--- Compute partials ----")
Y = zeros(24)
X = [0., 0., 70., 62., 70., 80., 82., 83., 76., 80., 80., 79., 78., 80., 78., 76., 79., 85., 79., 78., 71., 60., 54., 45.]
J = Y.*X'
#'

const f_tape = JacobianTape(f_tone_corrections_fwd!, Y, X)
const compiled_f_tape = compile(f_tape)

@time jacobian!(J, compiled_f_tape, x)
println(J)