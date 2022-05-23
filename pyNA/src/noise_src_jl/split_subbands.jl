function split_subbands(settings::PyObject, msap_in)
    
    # Number of time steps
    n_t = size(msap_in)[1]
    
    # Integer for subband calculation
    m = Int64((settings.N_b - 1) / 2)

#   # Divide msap_in by average
#     msap_proc = zeros(eltype(msap_in), (n_t, settings.N_f))        
#     msap_proc[findall(sum(msap_in, dims=2) .* ones(1, settings.N_f) .> 0)] = (msap_in ./ (sum(msap_in, dims=2) / settings.N_f))[findall(sum(msap_in, dims=2) .* ones(1, settings.N_f) .> 0)]
    
    # Calculate slope of spectrum
    # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 8-9
    u = zeros(eltype(msap_in), (n_t, settings.N_f, 1))
    v = zeros(eltype(msap_in), (n_t, settings.N_f, 1))
    u[:,2:end, 1]   = msap_in[:, 2:end] ./ msap_in[:,1:end-1] 
    v[:,2:end-1, 1] = msap_in[:, 3:end] ./ msap_in[:,2:end-1]
    u[:,1, 1] = v[:,1] = msap_in[:,2] ./ msap_in[:,1]
    u[:,end, 1] = v[:,end] = msap_in[:,end] ./ msap_in[:,end - 1]

    # Calculate constant A
    h = reshape(1:1:m, (1, 1, m))
    A = sum((u.^((h .- m .- 1)/settings.N_b) .+ v.^(h/settings.N_b)), dims=3)
    A = reshape(A + ones(eltype(msap_in), (n_t, settings.N_f)), (n_t, settings.N_f))
        
    h = reshape(1:1:settings.N_b, (1,1,settings.N_b))
    
    T = eltype(msap_in)
    msap_sb = zeros(T, (n_t, settings.N_f, settings.N_b))
    msap_sb[:,:,1:m] = ((msap_in./A) .* u .^ ((h .- m .-1) / settings.N_b))[:,:,1:m]
    msap_sb[:,:,m+1] = (msap_in./A)
    msap_sb[:,:,m+2:end] = ((msap_in./A) .* v .^ ((h .- m .-1) / settings.N_b))[:,:,m+2:end]
    msap_sb = transpose(reshape(permutedims(msap_sb, [3,2,1]), (settings.N_f*settings.N_b,n_t)))
    
    return msap_sb
end