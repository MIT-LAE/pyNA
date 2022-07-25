function smooth_max(k_smooth, x)

    # Compute lateral noise
    smooth_max = maximum(x) + 1/k_smooth.*log(sum(exp.(k_smooth*(x.-maximum(x)))))
    
    # Return
    return smooth_max
end