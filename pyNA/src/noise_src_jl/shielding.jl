function shielding(settings, data, j, observer)
    
    if settings.case_name in ["nasa_stca_standard", "stca_enginedesign_standard"] && settings.shielding == true
        if observer == "lateral"
            return data.shield_l[j, :]
        elseif observer == "flyover"
            return data.shield_f[j, :]
        elseif observer == "approach"
            return data.shield_a[j, :]
        end
    else
        return zeros(settings.N_f, )
    end

end