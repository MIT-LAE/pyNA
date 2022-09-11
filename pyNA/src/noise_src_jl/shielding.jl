function shielding(settings, data, j, i)
    
    if settings["case_name"] in ["nasa_stca_standard", "stca_enginedesign_standard"] && settings["shielding"] == true
        if settings["observer_lst"][i] == "lateral"
            return data.shield_l[j, :]
        elseif settings["observer_lst"][i] == "flyover"
            return data.shield_f[j, :]
        elseif settings["observer_lst"][i] == "approach"
            return data.shield_a[j, :]
        end
    else
        return zeros(settings["n_frequency_bands"], )
    end

end