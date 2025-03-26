import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import os

def create_prpsf_dict():

    #Initialize PRPSF dictory

    prpsf = {
        'Zstart': -1.0,
        'Zend': 1.0,
        'Zstep': 0.4,
        'Zindstart': 1,
        'Zindend': 6,
        'Zindstep': 1,
        'Beadcenter': None,
        'BeadXYshift': None,
        'BeadData': None,
        'DatadimX': None,
        'DatadimY': None,
        'DatadimZ': None,
        'PSFsize': None,
        'SubroiSize': None,
        'Pixelsize': None,
        'IterationNum': None,
        'IterationNumK': None,
        'ZernikeorderN': None,
        'OTFratioSize': None,
        'CCDoffset': None,
        'Gain': None,
        'FileDir': "",
        'FileName': "",
        'SaveDir': "",
        'Enableunwrap': None,
        'Zpos': None,
        'Z':None,
        # Private character
        'PhiC': None,
        'ZoC': None,
        'Phi': None,
        'k_r': None,
        'Zo': None,
        'NA_constrain': None,
        # Output character
        'PRstruct': {
            'NA': None,
            'Lambda': None,
            'RefractiveIndex': None,
            'Pupil': {'phase': None, 'mag': None},
            'Zernike_phase': None,
            'Zernike_mag': None,
            'Zernike_complex': None,
            'Fittedpupil': {'phase': None, 'mag': None},
            'SigmaX': None,
            'SigmaY': None,
            'Zernike_phaseinlambda': None
        },
        'PSFstruct': {
            'PRpsf': None,
            'ZKpsf': None,
            'Modpsf': None
        },
        'Mpsf_subroi': None,
        'Mpsf_extend': None
    }
    return prpsf

def prepdata(prpsf):

    # Data preparation, turn ADU count into photon numbers, avergae align time varation

    in_data = prpsf['BeadData']
    in_data = (in_data - prpsf['CCDoffset']) / prpsf['Gain']
    prpsf['BeadData'] = in_data
    prpsf['DatadimY'], prpsf['DatadimX'], prpsf['DatadimZ'] = in_data.shape
    prpsf['Mpsf_subroi'] = prpsf['BeadData']

def precomputeParam(prpsf):

    # precompute parameters, generate k-space operation image and saved

    XC, YC = np.meshgrid(np.arange(-prpsf['DatadimX'] / 2, prpsf['DatadimX'] / 2),
                         np.arange(-prpsf['DatadimY'] / 2, prpsf['DatadimY'] / 2))
    prpsf['PhiC'] = np.arctan2(YC, XC)
    prpsf['ZoC'] = np.sqrt(XC**2 + YC**2)

    X, Y = np.meshgrid(np.arange(-prpsf['PSFsize'] / 2, prpsf['PSFsize'] / 2),
                       np.arange(-prpsf['PSFsize'] / 2, prpsf['PSFsize'] / 2))
    prpsf['Zo'] = np.sqrt(X**2 + Y**2)
    scale = prpsf['PSFsize'] * prpsf['Pixelsize']
    prpsf['k_r'] = prpsf['Zo'] / scale
    prpsf['Phi'] = np.arctan2(Y, X)
    Freq_max = prpsf['PRstruct']['NA'] / prpsf['PRstruct']['Lambda']
    prpsf['NA_constrain'] = prpsf['k_r'] < Freq_max

    #  Initialize Zernike Polynomials
    prpsf['Z'] = create_zernike_state()
    prpsf['Z']['max_order'] = prpsf['ZernikeorderN']  # set bigget order number
    
    # Set polarized location parameters
    rho = np.where(prpsf['NA_constrain'], prpsf['k_r'] / Freq_max, 0)
    theta = np.where(prpsf['NA_constrain'], prpsf['Phi'], 0)
    
    # Generate basic matrix
    prpsf['Z'], ZM = generate_basis_matrix(
        prpsf['Z'], 
        rho, 
        theta, 
        prpsf['NA_constrain']
    )
    
    # Save basic matrix in dictionary
    prpsf['Z']['basis_matrix'] = ZM

    print(f"ZM 矩阵形状: {ZM.shape}")  # Expected (PSFsize, PSFsize, Zernike order)
    print(f"ZM 非零元素数量: {np.count_nonzero(ZM)}")

def genMpsf(prpsf):

    # Gengerate normalized psf for phase retrival

    R1 = prpsf['SubroiSize']
    R = prpsf['PSFsize']
    N = prpsf['DatadimZ']

    # Generate round mas
    X1, Y1 = np.meshgrid(np.arange(-R1 // 2, R1 // 2), np.arange(-R1 // 2, R1 // 2))
    circle_tmp = np.sqrt(X1**2 + Y1**2)
    circleF = np.where(circle_tmp <= R1 // 2 - 1, 1, 0)

    if R == R1:
        circleF = np.ones((R, R))

    # Calculate cutted region start and end index
    realsize0 = R1 // 2
    realsize1 = (R1 + 1) // 2
    startx = R // 2 - realsize0
    endx = startx + R1
    starty = R // 2 - realsize0
    endy = starty + R1

    MpsfL = np.zeros((R, R, N))

    for ii in range(N):
        Mpsfo = prpsf['Mpsf_subroi'][:, :, ii]
        
        # Calculate background value
        radius_mask = (circle_tmp <= (R1 // 2 - 1)) & (circle_tmp >= (R1 // 2 - 3))
        masks = [
            (X1 > 0) & (Y1 > 0) & radius_mask,  # First quardant
            (X1 <= 0) & (Y1 > 0) & radius_mask,  # Second quardant
            (X1 > 0) & (Y1 <= 0) & radius_mask,  # Third quardant
            (X1 <= 0) & (Y1 <= 0) & radius_mask  # Fourth quardant
        ]
        Edge = [Mpsfo[m].mean() + Mpsfo[m].std() for m in masks]
        bg = np.max(Edge)

        Fig2 = (Mpsfo - bg) * circleF
        Fig2 = np.clip(Fig2, 0, None) 

        tmp = np.zeros((R, R))
        tmp[starty:endy, startx:endx] = Fig2

        # Normalization
        total = np.sum(tmp)
        if total < 1e-12:
            tmp = np.ones_like(tmp) / (R * R)
        else:
            tmp = tmp / total

        MpsfL[:, :, ii] = tmp

    prpsf['Mpsf_extend'] = MpsfL
