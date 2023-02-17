using ReverseDiff
include("smooth_max.jl")


function lateral_attenuation(input_v::Union{Array, ReverseDiff.TrackedArray}, settings, x_obs::Array{Float64, 1})

    # input_v = [x, y, z]

    # Number of time steps
    n_t = Int64(size(input_v)[1]/3)

    # Compute elevation angle (with respect to the horizontal plane of the microphone) - similar to geometry module
    r_1 =  x_obs[1] .- input_v[1:n_t]
    r_2 =  x_obs[2] .- input_v[1*n_t+1:2*n_t]
    r_3 = -x_obs[3] .+ (input_v[2*n_t+1:3*n_t] .+ 4.)
    r = sqrt.(r_1.^2 + r_2.^2 + r_3.^2)
    n_vcr_a_3 = r_3 ./ r
    beta = asin.(n_vcr_a_3)*180/pi

    # Compute maximum elevation angle seen by the microphone using smooth maximum
    beta_max = smooth_max(beta, 50.)

    # Engine installation term [dB]
    if settings["lateral_attenuation_engine_mounting"] == "underwing"
        E_eng = 10 * log10.((0.0039 * cos.(beta_max*pi/180).^2 + sin.(beta_max*pi/180).^2).^0.062./(0.8786 * sin.(2 * beta_max*pi/180).^2 .+ cos.(2*beta_max*pi/180).^2))
    elseif settings["lateral_attenuation_engine_mounting"] == "fuselage"
        E_eng = 10 * log10.((0.1225 * cos.(beta_max*pi/180).^2 + sin.(beta_max*pi/180).^2).^0.329)
    elseif settings["lateral_attenuation_engine_mounting"] == "propeller"
        E_eng = 0.
    elseif settings["lateral_attenuation_engine_mounting"] == "none"
        E_eng = 0.
    end

    # Attenuation caused by ground and refracting-scattering effects [dB] (Note: beta is in degrees)
    if beta_max[1] < 50.
        A_grs = 1.137 .- 0.0229*beta_max .+ 9.72*exp.(-0.142*beta_max)
    elseif beta_max[1] < 0.
        throw(DomainError(beta_max, "beta_max is outside the valid domain (beta_max > 0)"))
    else
        A_grs = 0.
    end 

    # Over-ground attenuation [dB]
    if 0 <= x_obs[2] <= 914
        g = 11.83 * (1 - exp(-0.00274 * x_obs[2]))
    elseif x_obs[2] < 0.
        throw(DomainError(x_obs[2], "observer lateral position is outside the valid domain (x_obs[2] > 0)"))
    else
        g = 10.86  # 11.83*(1-exp(-0.00274*914))
    end

    # Overall lateral attenuation
    return E_eng - g*A_grs/10.86

end

lateral_attenuation_fwd = (x)->lateral_attenuation(x, settings, x_obs)