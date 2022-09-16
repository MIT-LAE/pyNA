function direct_propagation!(spl, x, settings)
    
    # x = [r, I_0]
    # y = spl
    
    # Impedance at the observer
    I_0_obs = 409.74

    # Apply spherical spreading and characteristic impedance effects to the MSAP
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 1
    dp = (settings["r_0"]^2 / x[1]^2) * (I_0_obs/x[2])
    spl .*= dp
    
end

direct_propagation_fwd! = (y,x) -> direct_propagation!(y, x, settings)