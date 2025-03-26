class PSF_pupil:
    def __init__(self, PRstruct):
        """
        Initialize the PSF_pupil object with the given PRstruct.

        Parameters:
        PRstruct (dict): A dictionary containing the following keys:
            - 'Pupil': A dictionary with 'mag' and 'phase' keys representing the pupil function.
            - 'NA': Numerical aperture.
            - 'Lambda': Wavelength, unit in microns.
            - 'RefractiveIndex': Refractive index of the immersion medium.
            - 'SigmaX': Sigma of Gaussian filter for OTF rescale in k-space, unit is 1/micron.
            - 'SigmaY': Sigma of Gaussian filter for OTF rescale in k-space, unit is 1/micron.
        """
        self.PRstruct = PRstruct
        self.PSFsize = PRstruct['Pupil']['mag'].shape[0]
        self.Xpos = None
        self.Ypos = None
        self.Zpos = None
        self.ZposMed = None
        self.nMed = None
        self.Boxsize = None
        self.Pixelsize = None

        self.Zo = None
        self.k_r = None
        self.k_z = None
        self.Phi = None
        self.NA_constrain = None
        self.Cos1 = None
        self.Cos3 = None

        self.PSFs = None
        self.IMMPSFs = None
        self.ScaledPSFs = None

    def precomputeParam(self):
      
        #Generate images for k-space operation and save them in precomputed parameters.
        X, Y = np.meshgrid(np.arange(-self.PSFsize / 2, self.PSFsize / 2),
                           np.arange(-self.PSFsize / 2, self.PSFsize / 2))
        self.Zo = np.sqrt(X**2 + Y**2)
        scale = self.PSFsize * self.Pixelsize
        self.k_r = self.Zo / scale
        self.Phi = np.arctan2(Y, X)
        n = self.PRstruct['RefractiveIndex']
        Freq_max = self.PRstruct['NA'] / self.PRstruct['Lambda']
        self.NA_constrain = self.k_r < Freq_max
        sin_theta3 = self.k_r * self.PRstruct['Lambda'] / n
        if self.nMed is None:
            self.nMed = n
        sin_theta1 = n / self.nMed * sin_theta3
        self.Cos1 = np.sqrt(1 - sin_theta1**2)
        self.Cos3 = np.sqrt(1 - sin_theta3**2)
        self.k_z = np.sqrt((n / self.PRstruct['Lambda'])**2 - self.k_r**2) * self.NA_constrain

    def genPSF(self):
        """
        Generate PSFs from the given pupil function.
        """
        N = len(self.Xpos)
        R = self.PSFsize
        Ri = self.Boxsize
        psfs = np.zeros((Ri, Ri, N))

        for ii in range(N):
            shiftphase = -self.k_r * np.cos(self.Phi) * self.Xpos[ii] * self.Pixelsize - \
                         self.k_r * np.sin(self.Phi) * self.Ypos[ii] * self.Pixelsize
            shiftphaseE = np.exp(-1j * 2 * np.pi * shiftphase)
            defocusphaseE = np.exp(2 * np.pi * 1j * self.Zpos[ii] * self.k_z)
            pupil_complex = self.PRstruct['Pupil']['mag'] * self.PRstruct['Pupil']['phase'] * shiftphaseE * defocusphaseE
            psfA = np.abs(fftshift(fft2(pupil_complex)))
            Fig2 = psfA**2
            realsize0 = int(np.floor(Ri / 2))
            realsize1 = int(np.ceil(Ri / 2))
            startx = -realsize0 + int(R / 2) + 1
            endx = realsize1 + int(R / 2)
            starty = -realsize0 + int(R / 2) + 1
            endy = realsize1 + int(R / 2)
            psfs[:, :, ii] = Fig2[startx:endx, starty:endy] / R**2

        self.PSFs = psfs

    def genIMMPSF(self):
        """
        Generate PSFs considering refractive index mismatch aberration.
        """
        n = self.PRstruct['RefractiveIndex']
        stagepos = self.Zpos[0]
        depth = stagepos * self.nMed / n
        N = len(self.Xpos)
        deltaH = depth * self.nMed * self.Cos1 - depth * n**2 / self.nMed * self.Cos3
        IMMphase = np.exp(2 * np.pi / self.PRstruct['Lambda'] * deltaH * self.NA_constrain * 1j)
        zMed = self.ZposMed
        zMed[zMed < -depth] = -depth

        R = self.PSFsize
        Ri = self.Boxsize
        psfs = np.zeros((Ri, Ri, N))

        for ii in range(N):
            defocusMed = np.exp(2 * np.pi / self.PRstruct['Lambda'] * self.nMed * zMed[ii] * self.Cos1 * 1j)
            shiftphase = -self.k_r * np.cos(self.Phi) * self.Xpos[ii] * self.Pixelsize - \
                         self.k_r * np.sin(self.Phi) * self.Ypos[ii] * self.Pixelsize
            shiftphaseE = np.exp(-1j * 2 * np.pi * shiftphase)
            pupil_complex = self.PRstruct['Pupil']['mag'] * self.PRstruct['Pupil']['phase'] * shiftphaseE * defocusMed * IMMphase
            psfA = np.abs(fftshift(fft2(pupil_complex)))
            Fig2 = psfA**2
            realsize0 = int(np.floor(Ri / 2))
            realsize1 = int(np.ceil(Ri / 2))
            startx = -realsize0 + int(R / 2) + 1
            endx = realsize1 + int(R / 2)
            starty = -realsize0 + int(R / 2) + 1
            endy = realsize1 + int(R / 2)
            psfs[:, :, ii] = Fig2[startx:endx, starty:endy] / R**2

        self.IMMPSFs = psfs

    def scalePSF(self):
        """
        Generate OTF rescaled PSFs.
        """
        otfobj = OTFrescale({
            'SigmaX': self.PRstruct['SigmaX'],
            'SigmaY': self.PRstruct['SigmaY'],
            'Pixelsize': self.Pixelsize,
            'PSFs': self.PSFs
        })
        otfobj.scaleRspace()
        self.ScaledPSFs = otfobj.Modpsfs
