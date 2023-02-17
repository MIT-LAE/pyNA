include("direct_propagation.jl")
include("atmospheric_absorption.jl")
include("lateral_attenuation.jl")
include("ground_effects.jl")
using ReverseDiff


function propagation!(spl::Array, x::Union{Array, ReverseDiff.TrackedArray}, settings, pyna_ip, f_sb::Array{Float64, 1}, x_obs::Array{Float64, 1})

    # x = [r, z, c_bar, rho_0, I_0, beta]
    # y = spl

    if settings["direct_propagation"]
        direct_propagation!(spl, vcat(x[1], x[5]), settings)
    end

    # Split mean-square acoustic pressure in frequency sub-bands
    if settings["n_frequency_subbands"] > 1
        spl_sb = zeros(eltype(x), settings["n_frequency_bands"]*settings["n_frequency_subbands"])
        split_subbands!(spl_sb, spl, settings)
    else
        spl_sb = spl
    end

    if settings["absorption"]
        atmospheric_absorption!(spl_sb, vcat(x[1], x[2]), settings, pyna_ip, f_sb)
    end

    # Apply ground effects on sub-bands
    if settings["ground_effects"]
        ground_effects!(spl_sb, vcat(x[1], x[6], x[3], x[4]), settings, pyna_ip, f_sb, x_obs)
    end
    
    # Recombine the mean-square acoustic pressure in the frequency sub-bands
    if settings["n_frequency_subbands"] > 1
        for i in 1:1:settings["n_frequency_bands"]
            spl[i] = sum(spl_sb[(i-1)*settings["n_frequency_subbands"]+1:i*settings["n_frequency_subbands"]])
        end
    end
    
end

propagation_fwd! = (y,x)->propagation!(y, x, settings, pyna_ip, f_sb, x_obs)