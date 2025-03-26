def phaseretrieve(obj):
    """
    Generate pupil function based on a phase retrieval algorithm described in the paper.

    Parameters:
    obj: Object containing PSF and OTF parameters.
    """
    z = obj['Zpos']
    zind = np.arange(0, len(z))
    N = len(zind)
    n = obj['PRstruct']['RefractiveIndex']
    Freq_max = obj['PRstruct']['NA'] / obj['PRstruct']['Lambda']
    NA_constrain = obj['k_r'] < Freq_max

    lambda_term = (n / obj['PRstruct']['Lambda']) ** 2
    k_r_sq = obj['k_r'] ** 2
    valid_mask = (lambda_term - k_r_sq) >= 0  
    k_z = np.sqrt(lambda_term - k_r_sq + 0j) * valid_mask

    pupil_mag = NA_constrain / np.sum(NA_constrain)  # Initial pupil function, normalization

    R = obj['PSFsize']
    MpsfA = np.zeros((R, R, N), dtype=complex)
    RpsfA_phase = np.zeros((R, R, N), dtype=complex)
    Rpupil_mag = np.zeros((R, R, N), dtype=complex)
    Rpupil_phase = np.zeros((R, R, N), dtype=complex)
    pupil_phase = np.ones((R, R), dtype=complex)

    for k in range(obj['IterationNum']):
        for j in range(N):
            defocus_phase = 2 * np.pi * z[zind[j]] * k_z
            pupil_complex = pupil_mag * np.exp(defocus_phase * 1j) * pupil_phase  # Apply defocus for each PSF section
            Fig1 = np.abs(fftshift(fft2(pupil_complex))) ** 2
            PSF0 = Fig1 / np.sum(Fig1)
            Mpsfo = obj['Mpsf_extend'][:, :, zind[j]]

            # At iteration number greater than IterationNumK, add previous retrieved PSF information in measured PSF
            if k > obj['IterationNumK']:
                Mask = (Mpsfo == 0)
                Mpsfo[Mask] = PSF0[Mask]

            RpsfA = fft2(pupil_complex)  # FFT each complex section
            RpsfA_phase[:, :, j] = RpsfA / np.maximum(np.abs(RpsfA), 1e-12)
            Fig2 = fftshift(np.sqrt(np.abs(Mpsfo)))
            MpsfA[:, :, j] = Fig2 / np.sum(Fig2)
            Rpupil = ifft2(MpsfA[:, :, j] * RpsfA_phase[:, :, j])  # Replace calculated magnitude with measured PSF

            Rpupil_mag[:, :, j] = np.abs(Rpupil)
            Rpupil_phase[:, :, j] = Rpupil / np.maximum(Rpupil_mag[:, :, j], 1e-12) * np.exp(-defocus_phase * 1j)

        # Generate pupil phase
        Fig5 = np.mean(Rpupil_phase, axis=2)
        denominator = np.maximum(np.abs(Fig5), 1e-12)
        pupil_phase = Fig5 / denominator

        # Generate pupil magnitude
        Fig3 = np.mean(Rpupil_mag, axis=2) * NA_constrain
        Fig4 = np.abs(Fig5) * Fig3  # Pupil magnitude before normalization
        Fig4 = Fig4 ** 2
        Fig4 = Fig4 / np.sum(Fig4)
        pupil_mag = np.sqrt(Fig4)  # Pupil magnitude after normalization

    # Generate phase retrieved PSF
    psf = np.zeros((R, R, len(z)))

    for j in range(len(z)):
        defocus_phase = 2 * np.pi * z[j] * k_z * 1j
        pupil_complex = pupil_mag * pupil_phase * np.exp(defocus_phase)
        Fig2 = np.abs(fftshift(fft2(pupil_complex))) ** 2
        psf[:, :, j] = Fig2 / R ** 2  # Normalized PSF

    # Save pupil function and PSF in PRstruct
    obj['PRstruct']['Pupil']['phase'] = pupil_phase
    obj['PRstruct']['Pupil']['mag'] = pupil_mag
    obj['PSFstruct']['PRpsf'] = psf
