function smooth_max(x, k_smooth)

    # Compute lateral noise
    return maximum(x) + 1/k_smooth.*log(sum(exp.(k_smooth*(x.-maximum(x)))))
    
end

function smooth_max!(y, x, k_smooth)

    # Compute lateral noise
    y .= maximum(x) + 1/k_smooth.*log(sum(exp.(k_smooth*(x.-maximum(x)))))
    
end

smooth_max_fwd! = (y, x)->smooth_max!(y, x, k_smooth)