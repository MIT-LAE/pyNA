using ReverseDiff

function atmospheric_absorption!(spl_sb::Array, x::Union{Array, ReverseDiff.TrackedArray}, settings, pyna_ip, f_sb::Array{Float64, 1})

    # x = [r, z]
    # y = spl_sb

    # Calculate average absorption factor between observer and source
    # Note: add 4 meters to the alitude of the aircraft (for engine height)
    alpha = pyna_ip.f_abs.((x[2] + 4).*ones(settings["n_frequency_subbands"]*settings["n_frequency_bands"],), f_sb)

    # Calculate absorption (convert dB to Np: 1dB is 0.115Np)
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 14
    spl_sb .*= exp.(-2 * 0.115 * alpha * (x[1] - settings["r_0"]))
end

atmospheric_absorption_fwd! = (y,x)-> atmospheric_absorption!(y, x, settings, pyna_ip, f_sb)