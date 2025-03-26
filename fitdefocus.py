def fitdefocus(obj, x, C4):

    z0 = x[0]  # Defocus paramter
    bg = x[1]  # Background parameter, should be close to 0
    
    PRstruct = obj['PRstruct']
    n = PRstruct['RefractiveIndex']
    NA = PRstruct['NA']
    Lambda = PRstruct['Lambda']
    
    # k_r uniform
    max_freq = NA / Lambda
    k_r_norm = obj['k_r'] / max_freq  # 归一化到[0,1]
    
    # constrained k_z calculation
    with np.errstate(invalid='ignore'):
        k_z = np.sqrt((n/Lambda)**2 - obj['k_r']**2) * (k_r_norm <= 1.0)
    k_z = np.nan_to_num(k_z, nan=0.0)
    
    # Accurate phase calculation
    phase = 2 * np.pi * z0 * k_z - bg
    defocus_phase = np.real(phase)
    
    #  Correct Zernike mode selection
    Z4 = obj['Z'].ZM[:, :, 3].copy()
    
    # Error calculation with regularization item
    error = C4 * Z4 - defocus_phase
    sse = np.sum(error**2) + 1e-6 * (z0**2 + bg**2)  # incase parameter error
    print(f"SSE={sse:.2e} | ")
    return sse
