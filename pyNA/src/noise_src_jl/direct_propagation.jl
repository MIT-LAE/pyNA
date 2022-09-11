function direct_propagation!(spl, settings, r, I_0)
    # Impedance at the observer
    I_0_obs = 409.74

    # Apply spherical spreading and characteristic impedance effects to the MSAP
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 1
    dp = (settings["r_0"]^2 / r^2) * (I_0_obs/I_0)
    @. spl *= dp
end

