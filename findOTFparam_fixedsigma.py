import numpy as np

def findOTFparam_fixedsigma(obj, sigma):
    """
    Find OTF parameters with fixed SigmaX and SigmaY of a Gaussian filter for OTF rescale.

    Parameters:
    obj: Object containing PSF and OTF parameters.
    sigma (float): Fixed value for SigmaX and SigmaY of the Gaussian filter.

    Returns:
    OTFparam (list): List containing [1, sigma, 0].
    """
    R = obj.PSFsize
    N = obj.DatadimZ
    OTFparam = [1, sigma, 0]

    # Generate Gaussian filter
    scale = R * obj.Pixelsize
    xx, yy = np.meshgrid(np.arange(-R / 2, R / 2), np.arange(-R / 2, R / 2))
    X = np.abs(xx) / scale
    Y = np.abs(yy) / scale
    fit_im = 1.0 * np.exp(-X**2 / (2 * sigma**2)) * np.exp(-Y**2 / (2 * sigma**2))

    # Generate Zernike fitted PSF modified by OTF rescale
    Mod_psf = np.zeros((R, R, N))
    for ii in range(N):
        Fig4 = obj.PSFstruct.PRpsf[:, :, ii]  # Changed by FX, from ZKpsf to PRpsf
        Fig4 = Fig4 / np.sum(Fig4)
        Mod_OTF = np.fft.fftshift(np.fft.ifft2(Fig4)) * fit_im
        Fig5 = np.abs(np.fft.fft2(Mod_OTF))
        Mod_psf[:, :, ii] = Fig5

    # Save SigmaX and SigmaY in PRstruct
    obj.PRstruct.SigmaX = sigma
    obj.PRstruct.SigmaY = sigma

    # Save modified PSF in PSFstruct
    obj.PSFstruct.Modpsf = Mod_psf

    return OTFparam
