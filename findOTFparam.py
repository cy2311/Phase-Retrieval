import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from scipy.optimize import minimize

def findOTFparam(obj):
    """
    Find SigmaX and SigmaY of a Gaussian filter for OTF rescale.

    Parameters:
    obj: Object containing PSF and OTF parameters.

    Returns:
    OTFparam (list): List containing [I, sigma, bg].
    """
    R = obj['PSFsize']
    z = obj['Zpos']
    N = obj['DatadimZ']

    # Find the index of the closest Z position to zero
    ind = np.argmin(np.abs(z))

    # Define the region of interest for the OTF ratio
    realsize0 = int(np.floor(obj['OTFratioSize'] / 2))
    realsize1 = int(np.ceil(obj['OTFratioSize'] / 2))
    starty = -realsize0 + R // 2
    endy = realsize1 + R // 2
    startx = -realsize0 + R // 2
    endx = realsize1 + R // 2

    # Compute the measured OTF and phase-retrieved OTF
    mOTF = fftshift(ifft2(obj['Mpsf_extend'][:, :, ind]))  # Measured OTF
    rOTF = fftshift(ifft2(obj['PSFstruct']['PRpsf'][:, :, ind]))  # Phase-retrieved OTF

    # Compute the OTF ratio
    tmp = np.abs(mOTF) / np.abs(rOTF)
    ratio = tmp[startx:endx, starty:endy]

    # Fit the OTF ratio with a 2D Gaussian
    fit_param = [1, 2, 0]
    I, sigma, bg, fit_im = GaussRfit(obj, fit_param, ratio)
    OTFparam = [I, sigma, bg]

    # Generate Zernike fitted PSF modified by OTF rescale
    Mod_psf = np.zeros((R, R, N))
    for ii in range(N):
        Fig4 = obj['PSFstruct']['PRpsf'][:, :, ii]
        Fig4 = Fig4 / np.sum(Fig4)
        Mod_OTF = fftshift(ifft2(Fig4)) * fit_im
        Fig5 = np.abs(fft2(Mod_OTF))
        Mod_psf[:, :, ii] = Fig5

    # Save SigmaX and SigmaY in PRstruct
    obj['PRstruct']['SigmaX'] = sigma
    obj['PRstruct']['SigmaY'] = sigma

    # Save modified PSF in PSFstruct
    obj['PSFstruct']['Modpsf'] = Mod_psf

    return OTFparam


def GaussRfit(obj, startpoint, input_im):
    """
    Fit a 2D Gaussian to the input image.

    Parameters:
    obj: Object containing PSF and OTF parameters.
    startpoint (list): Initial guess for the Gaussian parameters [I, sigma, bg].
    input_im (numpy.ndarray): Input image to fit.

    Returns:
    I (float): Amplitude of the Gaussian.
    sigma (float): Standard deviation of the Gaussian.
    bg (float): Background value.
    fit_im (numpy.ndarray): Fitted Gaussian image.
    """
    R1 = obj['OTFratioSize']
    R = obj['PSFsize']

    # Optimize the Gaussian parameters
    result = minimize(
        lambda x: Gauss2(x, input_im, R1, R, obj['Pixelsize']),
        startpoint,
        method='Nelder-Mead',
        options={'maxiter': 50, 'disp': False}
    )
    estimate = result.x

    I = estimate[0]
    sigma = min(estimate[1], 5)  # Restrict sigma to be less than 5
    bg = 0

    # Generate the fitted Gaussian image
    scale = R * obj['Pixelsize']
    xx, yy = np.meshgrid(np.arange(-R // 2, R // 2), np.arange(-R // 2, R // 2))
    X = np.abs(xx) / scale
    Y = np.abs(yy) / scale
    fit_im = I * np.exp(-X**2 / (2 * sigma**2)) * np.exp(-Y**2 / (2 * sigma**2)) + bg

    return I, sigma, bg, fit_im


def Gauss2(x, input_im, R1, R, pixelsize):
    """
    Compute the sum of squared errors between the input image and a 2D Gaussian model.

    Parameters:
    x (list): Gaussian parameters [I, sigma, bg].
    input_im (numpy.ndarray): Input image.
    R1 (int): Size of the input image.
    R (int): Size of the output image.
    pixelsize (float): Pixel size.

    Returns:
    sse (float): Sum of squared errors.
    """
    I = x[0]
    sigma = x[1]
    bg = 0

    # Generate the Gaussian model
    xx, yy = np.meshgrid(np.arange(-R1 // 2, R1 // 2), np.arange(-R1 // 2, R1 // 2))
    scale = R * pixelsize
    X = np.abs(xx) / scale
    Y = np.abs(yy) / scale
    Model = I * np.exp(-X**2 / (2 * sigma**2)) * np.exp(-Y**2 / (2 * sigma**2)) + bg

    # Compute the sum of squared errors
    sse = np.sum((Model - input_im) ** 2)
    print(f"SSE={sse:.2e} | ")
    return sse
