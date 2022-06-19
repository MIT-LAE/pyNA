include("pnl.jl")
include("tone_corrections.jl")


function f_pnlt(pyna_ip, settings, f, spl)

    # Compute noy
    N = pyna_ip.f_noy.(spl, f)

    # Compute pnl
    pnl = f_pnl(N)
    
    # Compute the correction factor C
    c_max = f_tone_corrections(settings, spl)
    
    # Compute PNLT    
    pnlt = pnl + c_max

    return pnlt
end