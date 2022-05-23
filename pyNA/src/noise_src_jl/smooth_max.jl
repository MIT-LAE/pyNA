function smooth_max(k_smooth::Float64, level_sideline)

    # Compute lateral noise
    smooth_max = maximum(level_sideline) + 1/k_smooth.*log(sum(exp.(k_smooth*(level_sideline.-maximum(level_sideline)))))
    
    # Return
    return smooth_max
end