function shielding(settings, data, j, observer)
    
    if settings.case_name in ["nasa_stca_standard", "stca_enginedesign_standard"] && settings.shielding == true
        if observer == "lateral"
            shield = data.shield_l[j, :]
        elseif observer == "flyover"
            shield = data.shield_f[j, :]
        elseif observer == "approach"
            shield = data.shield_a[j, :]
        end
    else
        shield = zeros(1, settings.N_f)
    end

    return shield
end