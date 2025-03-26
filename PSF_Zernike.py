import numpy as np
from scipy.fftpack import fft2, fftshift
from typing import Dict, Optional, Tuple

class PSF_zernike:
    def __init__(self, PRstruct: Dict):

        self.PRstruct = PRstruct.copy()
        self.PSFsize = PRstruct.get('PSFsize', 128)
        self.Boxsize = PRstruct.get('Boxsize', 128)
        
        self.Pupil = {'phase': None, 'mag': None}
        if 'Pupil' in PRstruct:
            self.Pupil.update(PRstruct['Pupil'])
        
        self.Xpos = np.array([0.0])
        self.Ypos = np.array([0.0])
        self.Zpos = np.array([0.0])
        self.ZposMed = None
        self.nMed = None
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
        
        self._zernike_state = self._init_zernike_state()

    def _init_zernike_state(self) -> Dict:
        return {
            'ordering': 'Wyant',
            'w_abs_err': 1e-10,
            'w_rel_err': 1e-10,
            'basis_matrix': None,
            'max_order': -1,
            'max_index': -1,
            'coefficients': [],
            'normalization_radius': 1.0
        }

    def precomputeParam(self):
        R = self.PSFsize
        x = np.linspace(-R/2, R/2-1, R)
        y = np.linspace(-R/2, R/2-1, R)
        X, Y = np.meshgrid(x, y, indexing='xy')
        
        self.Zo = np.sqrt(X**2 + Y**2)
        scale = R * self.Pixelsize
        self.k_r = self.Zo / scale
        self.Phi = np.arctan2(Y, X)
        
        n = self.PRstruct['RefractiveIndex']
        freq_max = self.PRstruct['NA'] / self.PRstruct['Lambda']
        self.NA_constrain = (self.k_r < freq_max).astype(float)
        
        self.k_z = np.sqrt((n/self.PRstruct['Lambda'])**2 - self.k_r**2 + 0j) * self.NA_constrain
        
        if self.nMed is not None:
            sin_theta3 = self.k_r * self.PRstruct['Lambda'] / n
            sin_theta1 = n / self.nMed * sin_theta3
            self.Cos1 = np.sqrt(1 - sin_theta1**2 + 0j)
            self.Cos3 = np.sqrt(1 - sin_theta3**2 + 0j)

    def _compute_zernike_basis(self) -> np.ndarray:
        rho = self.k_r * self.NA_constrain / (self.PRstruct['NA']/self.PRstruct['Lambda'])
        theta = self.Phi * self.NA_constrain
        
        ZN = int(np.sqrt(len(self.PRstruct['Zernike_phase']))) - 1
        self._zernike_state['max_order'] = ZN
        
        self._zernike_state, Z = self._generate_zernike_basis(
            self._zernike_state, 
            rho, 
            theta,
            self.NA_constrain
        )
        return Z

    def setPupil(self):
        if 'Pupil' not in self.PRstruct:
            raise ValueError("PRstruct中缺少Pupil数据")
        
        # deepcopy induce quote cite problem
        self.Pupil['phase'] = np.array(self.PRstruct['Pupil']['phase'], copy=True)
        
        # 2-value magnitude
        mag = np.array(self.PRstruct['Pupil']['mag'], copy=True)
        mag[mag > 0] = 1.0
        self.Pupil['mag'] = mag
        
        #print(f"[setPupil] phase范围: {np.min(self.Pupil['phase']):.3f}~{np.max(self.Pupil['phase']):.3f}")
        #print(f"[setPupil] mag非零像素: {np.sum(self.Pupil['mag'] > 0)}/{self.Pupil['mag'].size}")

    @staticmethod
    def _generate_zernike_basis(
        state: Dict,
        rho: np.ndarray,
        theta: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[Dict, np.ndarray]:
        max_order = state['max_order']
        if max_order == -1:
            raise ValueError("Max order not set")
        
        coefficients = []
        for n in range(max_order + 1):
            n_coeffs = []
            for m in range(n + 1):
                c = np.zeros(n - m + 1)
                for k in range(len(c)):
                    start = n - m - k + 1
                    end = 2*n - m - k
                    num = (-1)**k * np.prod(np.arange(start, end+1)) if start <= end else 1.0
                    den = np.math.factorial(k) * np.math.factorial(n - k)
                    c[k] = num / den
                n_coeffs.append(c)
            coefficients.append(n_coeffs)
        
        state['coefficients'] = coefficients
        R, C = rho.shape
        n_basis = (max_order + 1)**2
        Z = np.zeros((R, C, n_basis))
        
        if mask is None:
            mask = np.ones_like(rho, dtype=bool)
        
        max_power = 2 * max_order
        rho_pows = np.zeros((max_power + 1, R, C))
        rho_pows[0] = 1.0
        for i in range(1, max_power + 1):
            rho_pows[i] = rho_pows[i-1] * rho
        
        basis_idx = 0
        Z[..., 0] = mask.astype(float)  # Piston项
        basis_idx += 1
        
        for n in range(1, max_order + 1):
            for m in range(n, -1, -1):
                if m >= len(coefficients[n]):
                    continue
                
                c = coefficients[n][m]
                exponents = np.arange(2*n - m, m - 1, -2)
                
                radial = np.zeros_like(rho)
                for k, exp in enumerate(exponents):
                    radial += c[k] * rho_pows[exp]
                
                if m == 0:
                    Z[..., basis_idx] = radial * mask
                    basis_idx += 1
                else:
                    Z[..., basis_idx] = radial * np.cos(m * theta) * mask
                    Z[..., basis_idx+1] = radial * np.sin(m * theta) * mask
                    basis_idx += 2
                
                if basis_idx >= n_basis:
                    break
        
        state['basis_matrix'] = Z
        return state, Z

    def genPupil(self):
        Z = self._compute_zernike_basis()
        
        # Phase part
        pupil_phase = np.zeros((self.PSFsize, self.PSFsize))
        for k, coeff in enumerate(self.PRstruct['Zernike_phase']):
            pupil_phase += Z[..., k] * coeff
        
        # Magnitude part
        pupil_mag = np.zeros_like(pupil_phase)
        for k, coeff in enumerate(self.PRstruct['Zernike_mag']):
            pupil_mag += Z[..., k] * coeff
        
        # Apply NA constrain
        pupil_phase = np.exp(1j * pupil_phase) * self.NA_constrain
        pupil_mag = pupil_mag * self.NA_constrain
        
        self.Pupil['phase'] = pupil_phase
        self.Pupil['mag'] = pupil_mag
        self.PRstruct['Pupil'] = self.Pupil.copy()


    def genPSF(self):

        N = len(self.Xpos)
        R = self.PSFsize
        Ri = self.Boxsize
        psfs = np.zeros((Ri, Ri, N))

        for ii in range(N):
            shiftphase = (-self.k_r * np.cos(self.Phi) * self.Xpos[ii] * self.Pixelsize -
                          self.k_r * np.sin(self.Phi) * self.Ypos[ii] * self.Pixelsize)
            shiftphaseE = np.exp(-1j * 2 * np.pi * shiftphase)
            defocusphaseE = np.exp(2 * np.pi * 1j * self.Zpos[ii] * self.k_z)
            pupil_complex = (self.Pupil['mag'] * np.exp(self.Pupil['phase'] * 1j) * shiftphaseE * defocusphaseE)
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

    def genPSF_2(self):

        self.Xpos = np.atleast_1d(self.Xpos)
        self.Ypos = np.atleast_1d(self.Ypos)
        self.Zpos = np.atleast_1d(self.Zpos)
        N = len(self.Xpos)
        
        if self.Pupil['mag'].shape != (self.PSFsize, self.PSFsize) or \
           self.Pupil['phase'].shape != (self.PSFsize, self.PSFsize):
            raise ValueError(f"Pupil 函数尺寸不匹配: 期望 ({self.PSFsize}, {self.PSFsize}), 实际 {self.Pupil['mag'].shape} 和 {self.Pupil['phase'].shape}")
        
        Ri = self.Boxsize
        psfs = np.zeros((Ri, Ri, N))
        
        for ii in range(N):
            shiftphase = (-self.k_r * np.cos(self.Phi) * self.Xpos[ii] * self.Pixelsize -
                          self.k_r * np.sin(self.Phi) * self.Ypos[ii] * self.Pixelsize)
            shiftphaseE = np.exp(-1j * 2 * np.pi * shiftphase)
            defocusphaseE = np.exp(2j * np.pi * self.Zpos[ii] * self.k_z)
            pupil_complex = (self.Pupil['mag'].astype(np.complex128) *
                            np.exp(1j * self.Pupil['phase'].astype(np.complex128)) *
                            shiftphaseE * defocusphaseE)
            psf_fft = np.fft.fft2(pupil_complex)
            psf_shifted = np.fft.fftshift(psf_fft)
            psfA = np.abs(psf_shifted)
            start = (self.PSFsize - self.Boxsize) // 2
            end = start + self.Boxsize
            psf_cropped = psfA[start:end, start:end] / self.PSFsize**2
            psfs[:, :, ii] = psf_cropped
        
        self.PSFs = psfs
        return self

    def genIMMPSF(self):
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
            shiftphase = (-self.k_r * np.cos(self.Phi) * self.Xpos[ii] * self.Pixelsize -
                          self.k_r * np.sin(self.Phi) * self.Ypos[ii] * self.Pixelsize)
            shiftphaseE = np.exp(-1j * 2 * np.pi * shiftphase)
            pupil_complex = (self.Pupil['mag'] * np.exp(self.Pupil['phase'] * 1j) *
                            shiftphaseE * defocusMed * IMMphase)
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

    def scalePSF(self, psftype='normal'):
        otfobj = OTFrescale({
            'SigmaX': self.PRstruct['SigmaX'],
            'SigmaY': self.PRstruct['SigmaY'],
            'Pixelsize': self.Pixelsize,
            'PSFs': self.PSFs if psftype == 'normal' else self.IMMPSFs
        })
        otfobj.scaleRspace()
        self.ScaledPSFs = otfobj.Modpsfs
