import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d

class OTFrescale:
    def __init__(self, params):
        """
        Initialize the OTFrescale object with the given parameters.

        Parameters:
        params (dict): A dictionary containing the following keys:
            - 'PSFs': Input PSFs, a 3D numpy array with the third dimension being the number of PSFs.
            - 'Pixelsize': Pixel size at the sample plane, in microns.
            - 'SigmaX': Sigma of Gaussian filter for OTF rescale in k-space, in 1/micron.
            - 'SigmaY': Sigma of Gaussian filter for OTF rescale in k-space, in 1/micron.
        """
        self.PSFs = params['PSFs']
        self.Pixelsize = params['Pixelsize']
        self.SigmaX = params['SigmaX']
        self.SigmaY = params['SigmaY']
        self.Modpsfs = None

    def scaleKspace(self):
        """
        Apply OTF rescale in k-space.
        """
        sz = self.PSFs.shape
        R = sz[0]
        N = 1 if len(sz) == 2 else sz[2]
        scale = R * self.Pixelsize

        xx, yy = np.meshgrid(np.arange(-R/2, R/2), np.arange(-R/2, R/2))
        X = np.abs(xx) / scale
        Y = np.abs(yy) / scale

        I = 1
        bg = 0
        gauss_k = I * np.exp(-X**2 / (2 * self.SigmaX**2)) * np.exp(-Y**2 / (2 * self.SigmaY**2)) + bg

        Mod_psf = np.zeros(sz, dtype=np.complex128)
        for ii in range(N):
            Fig1 = self.PSFs[:, :, ii] if N > 1 else self.PSFs
            Mod_OTF = fftshift(ifft2(Fig1)) * gauss_k
            Fig2 = np.abs(fft2(Mod_OTF))
            if N > 1:
                Mod_psf[:, :, ii] = Fig2
            else:
                Mod_psf = Fig2

        self.Modpsfs = Mod_psf

    def scaleRspace(self):
        """
        Apply OTF rescale in real space.
        """
        sz = self.PSFs.shape
        R = sz[0]
        N = 1 if len(sz) == 2 else sz[2]
        cropsize = min(29, R)

        sigmaXr = 1 / (2 * np.pi * self.SigmaX)
        sigmaYr = 1 / (2 * np.pi * self.SigmaY)

        X, Y = np.meshgrid(np.arange(-R/2, R/2), np.arange(-R/2, R/2))
        xx = X * self.Pixelsize
        yy = Y * self.Pixelsize

        I = 1
        tmp = I * 2 * np.pi * self.SigmaX * self.SigmaY * np.exp(-xx**2 / (2 * sigmaXr**2)) * np.exp(-yy**2 / (2 * sigmaYr**2))

        realsize0 = int(np.floor(cropsize / 2))
        realsize1 = int(np.ceil(cropsize / 2))
        starty = int(np.floor(-realsize0 + R/2 + 1))
        endy = int(np.floor(realsize1 + R/2))
        startx = int(np.floor(-realsize0 + R/2 + 1))
        endx = int(np.floor(realsize1 + R/2))

        gauss_r = tmp[starty:endy, startx:endx]
        gauss_r = gauss_r / np.sum(gauss_r)

        Mod_psf = np.zeros(sz)
        for ii in range(N):
            Fig1 = self.PSFs[:, :, ii] if N > 1 else self.PSFs
            Fig1 = np.squeeze(Fig1)  
            if Fig1.ndim != 2:
                raise ValueError(f"PSF slice must be 2-D, but got shape {Fig1.shape}")
            Mod_psf[:, :, ii] = convolve2d(Fig1, gauss_r, mode='same')

        self.Modpsfs = Mod_psf
