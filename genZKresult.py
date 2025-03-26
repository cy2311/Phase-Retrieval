def genZKresult(obj):
    """
    Expand the phase retrieved pupil function into Zernike polynomials.
    Uses the dictionary-based Zernike interface.
    """
    pupil_phase = obj['PRstruct']['Pupil']['phase']
    pupil_mag = obj['PRstruct']['Pupil']['mag']
    n = obj['PRstruct']['RefractiveIndex']
    R = obj['PSFsize']
    Z_N = obj['ZernikeorderN']

    if np.isrealobj(pupil_phase):
        R_aber = pupil_phase * obj['NA_constrain']
    else:
        R_aber = np.angle(pupil_phase) * obj['NA_constrain']

    U = pupil_mag * np.cos(R_aber) * obj['NA_constrain']
    V = pupil_mag * np.sin(R_aber) * obj['NA_constrain']
    complex_Mag = U + 1j * V

    # 使用新的拟合接口 - 修改为使用fit_zernike函数
    CN_complex, pupil_complexfit = fit_zernike(obj['Z'], complex_Mag, Z_N)
    CN_phase, pupil_phasefit = fit_zernike(obj['Z'], R_aber, Z_N)
    CN_mag, pupil_magfit = fit_zernike(obj['Z'], pupil_mag, Z_N)

    # 解包裹处理
    if obj.get('Enableunwrap', False):
        from scipy.ndimage import convolve
        A = np.ones((5, 5)) / 25  # Gaussian filter
        tmp = convolve(pupil_phase, A, mode='constant')
        R_aber = np.angle(tmp) * obj['NA_constrain']
        uwR_aber = unwrapPupil(R_aber)
        CN_phase, pupil_phasefit = fit_zernike(obj['Z'], uwR_aber, Z_N)

    # Generate Zernike fitted PSF
    z = obj['Zpos']
    N = len(z)
    zernike_psf = np.zeros((R, R, N))
    k_z = np.sqrt((n / obj['PRstruct']['Lambda']) ** 2 - obj['k_r'] ** 2) * obj['NA_constrain']

    for j in range(N):
        defocus_phase = 2 * np.pi * z[j] * k_z * 1j
        pupil_complex = pupil_magfit * np.exp(pupil_phasefit * 1j) * np.exp(defocus_phase)
        psfA = np.abs(fftshift(fft2(pupil_complex)))
        Fig2 = psfA ** 2
        zernike_psf[:, :, j] = Fig2 / R ** 4

    # Save results
    obj['PSFstruct']['ZKpsf'] = zernike_psf
    obj['PRstruct']['Zernike_phase'] = CN_phase
    obj['PRstruct']['Zernike_phaseinlambda'] = CN_phase / (2 * np.pi)
    obj['PRstruct']['Zernike_mag'] = CN_mag
    obj['PRstruct']['Zernike_complex'] = CN_complex
    obj['PRstruct']['Fittedpupil'] = {
        'complex': pupil_complexfit,
        'phase': pupil_phasefit,
        'mag': pupil_magfit
    }
