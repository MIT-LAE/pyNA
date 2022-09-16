include("pnl.jl")
include("tone_corrections.jl")
include("smooth_max.jl")


function f_pnlt(spl, settings, pyna_ip, f)

    # Compute noy
    N = pyna_ip.f_noy.(spl, f)

    # Compute pnl
    pnl = f_pnl(N)
    
    # Compute the correction factor C
    C = zeros(eltype(spl), settings["n_frequency_bands"])
    f_tone_corrections!(C, spl, settings)

    # Compute the largest of the tone correction
    if settings["tones_under_800Hz"]
        c_max = smooth_max(C, 50.)
    else
        c_max = smooth_max(C[14:end], 50.)
    end

    # Compute PNLT       
    pnlt = pnl + c_max

    if pnlt < 1e-99
        pnlt = 1e-99
    end

    return pnlt

end

function f_pnlt!(pnlt, spl, settings, pyna_ip, f)

    # Compute noy
    N = pyna_ip.f_noy.(spl, f)

    # Compute pnl
    pnl = zeros(eltype(spl), 1)
    f_pnl!(pnl, N)
    
    # Compute the correction factor C
    C = zeros(eltype(spl), settings["n_frequency_bands"])
    f_tone_corrections!(C, spl, settings)

    # Compute the largest of the tone correction
    c_max = zeros(eltype(spl), 1)
    if settings["tones_under_800Hz"]
        smooth_max!(c_max, C, 50.)
    else
        smooth_max!(c_max, C[14:end], 50.)
    end

    # Compute PNLT       
    pnlt .= pnl + c_max

end

f_pnlt_fwd! = (y,x)->f_pnlt!(y, x, settings, pyna_ip, f)