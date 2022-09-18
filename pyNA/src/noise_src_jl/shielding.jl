function shielding(settings, data, j::Int64, i::Int64)

    # j timestep
    # i observer number

    if settings["observer_lst"][i] == "lateral"
        return data.shield_l[j, :]
    elseif settings["observer_lst"][i] == "flyover"
        return data.shield_f[j, :]
    elseif settings["observer_lst"][i] == "approach"
        return data.shield_a[j, :]
    end
    
end