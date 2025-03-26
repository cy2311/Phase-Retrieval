import numpy as np
from scipy.fft import fft2, ifft2, fftshift

def unwrapPupil(obj, pupil):
    """
    Unwrap the phase of the pupil function.

    Parameters:
    obj: Object containing PSF and OTF parameters.
    pupil (numpy.ndarray): Input pupil phase.

    Returns:
    pupilUnwrapped (numpy.ndarray): Unwrapped pupil phase.
    """
    # Initialize
    Nxyspace = obj.PSFsize
    support = obj.NA_constrain

    # Modify pupil by repeating edge values
    nvalidRow = 0
    validRows = []
    for nrow in range(Nxyspace):
        ncol1 = np.where(support[nrow, :] == 1)[0][0] if np.any(support[nrow, :] == 1) else None
        ncol2 = np.where(support[nrow, :] == 1)[0][-1] if np.any(support[nrow, :] == 1) else None
        if ncol1 is not None and abs(obj.Phi[nrow, ncol2]) <= np.pi / 4:
            nvalidRow += 1
            validRows.append(nrow)
            pupil[nrow, :ncol1] = pupil[nrow, ncol1]
            pupil[nrow, ncol2:] = pupil[nrow, ncol2]

    nvalidCol = 0
    validCols = []
    for ncol in range(Nxyspace):
        nrow1 = np.where(support[:, ncol] == 1)[0][0] if np.any(support[:, ncol] == 1) else None
        nrow2 = np.where(support[:, ncol] == 1)[0][-1] if np.any(support[:, ncol] == 1) else None
        if nrow1 is not None and abs(obj.Phi[nrow1, ncol]) >= np.pi / 4 and abs(obj.Phi[nrow1, ncol]) <= 3 * np.pi / 4:
            nvalidCol += 1
            validCols.append(ncol)
            pupil[:nrow1, ncol] = pupil[nrow1, ncol]
            pupil[nrow2:, ncol] = pupil[nrow2, ncol]

    # Extend pupil values towards corners
    pupil[:validRows[0], validCols[-1]:] = pupil[validRows[0], validCols[-1]]  # 1st quadrant
    pupil[:validRows[0], :validCols[0]] = pupil[validRows[0], validCols[0]]  # 2nd quadrant
    pupil[validRows[-1]:, :validCols[0]] = pupil[validRows[-1], validCols[0]]  # 3rd quadrant
    pupil[validRows[-1]:, validCols[-1]:] = pupil[validRows[-1], validCols[-1]]  # 4th quadrant

    # Mirror pupil along edges to form 4 quadrants
    periodicPupil = np.zeros((2 * Nxyspace, 2 * Nxyspace), dtype=complex)
    periodicPupil[:Nxyspace, :Nxyspace] = pupil  # 2nd quadrant
    periodicPupil[:Nxyspace, Nxyspace:] = np.fliplr(periodicPupil[:Nxyspace, :Nxyspace])  # 1st quadrant
    periodicPupil[Nxyspace:, :] = np.flipud(periodicPupil[:Nxyspace, :])  # 3rd and 4th quadrant

    # Find derivatives numerically and wrap to values within [-pi, pi]
    xDeriv = np.roll(periodicPupil, -1, axis=1) - periodicPupil
    xDeriv = np.angle(np.exp(1j * xDeriv))  # Wrap
    yDeriv = np.roll(periodicPupil, -1, axis=0) - periodicPupil
    yDeriv = np.angle(np.exp(1j * yDeriv))  # Wrap

    # Calculate B3
    xDerivShiftm = np.roll(xDeriv, 1, axis=1)
    yDerivShiftm = np.roll(yDeriv, 1, axis=0)
    rho = (xDeriv - xDerivShiftm) + (yDeriv - yDerivShiftm)

    # Calculate B5
    n, m = np.meshgrid(np.arange(2 * Nxyspace), np.arange(2 * Nxyspace))
    n = n - Nxyspace
    m = m - Nxyspace
    nominator = 2 * np.cos(np.pi * m / Nxyspace) + 2 * np.cos(np.pi * n / Nxyspace) - 4
    oi = fft2(rho) / fftshift(nominator)
    oi[np.isinf(oi)] = 0
    oi[np.isnan(oi)] = 0
    pupilUnwrapped = ifft2(oi)
    pupilUnwrapped = pupilUnwrapped[:Nxyspace, :Nxyspace] * support

    # Save unwrapped pupil phase in PRstruct
    obj.PRstruct['Pupil']['uwphase'] = pupilUnwrapped

    return pupilUnwrapped
