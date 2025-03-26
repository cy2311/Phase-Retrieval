import numpy as np
from copy import deepcopy

def gen_initPupil(empupil, prev_probj=None):

    R = 128
    n_basis = empupil['Zernike_sz']
    
    if prev_probj is None:
        probj = {
            'PRstruct': {
                'NA': empupil['NA'],
                'Lambda': empupil['Lambda'],
                'RefractiveIndex': empupil['nMed'],
                'PSFsize': R,
                'Boxsize': R,
                'SigmaX': empupil['blur_sigma'],
                'SigmaY': empupil['blur_sigma'],
                'Pupil': {
                    'phase': np.zeros((R, R)),
                    'mag': np.zeros((R, R))
                },
                'Zernike_phase': np.zeros(n_basis),
                'Zernike_mag': np.zeros(n_basis)
            },
            'Pixelsize': empupil['Pixelsize'],
            'PSFsize': R,
            'nMed': empupil['nMed'],
            'Xpos': np.array([0.0]),
            'Ypos': np.array([0.0]),
            'Zpos': np.array([0.0])
        }
        
        probj['PRstruct']['Zernike_phase'][4:25] = empupil['init_z']
        probj['PRstruct']['Zernike_mag'][0] = 1.0
    else:
        probj = deepcopy(prev_probj)
        
        probj['PRstruct']['SigmaX'] = empupil['blur_sigma']
        probj['PRstruct']['SigmaY'] = empupil['blur_sigma']
        probj['Pixelsize'] = empupil['Pixelsize']
        probj['nMed'] = empupil['nMed']
    
    print('Generating initial pupil' if prev_probj is None else 'Updating pupil for iteration')
    
    temp_psf = PSF_zernike(probj['PRstruct'])
    temp_psf.Pixelsize = probj['Pixelsize']
    temp_psf.PSFsize = probj['PSFsize']
    temp_psf.nMed = probj['nMed']
    temp_psf.Xpos = probj['Xpos']
    temp_psf.Ypos = probj['Ypos']
    temp_psf.Zpos = probj['Zpos']
    
    temp_psf.precomputeParam()
    temp_psf._compute_zernike_basis()
    temp_psf.genPupil()
    
    probj['PRstruct']['Pupil'] = {
        'phase': temp_psf.Pupil['phase'].copy(),
        'mag': temp_psf.Pupil['mag'].copy()
    }
    probj['PRstruct']['Zernike_phase'] = temp_psf.PRstruct['Zernike_phase'].copy()
    probj['PRstruct']['Zernike_mag'] = temp_psf.PRstruct['Zernike_mag'].copy()
    
    probj.update({
        'Zo': temp_psf.Zo,
        'k_r': temp_psf.k_r,
        'k_z': temp_psf.k_z,
        'Phi': temp_psf.Phi,
        'NA_constrain': temp_psf.NA_constrain,
        'Z': {
            'ZM': temp_psf._zernike_state['basis_matrix'],
            'Zindex': np.arange(n_basis)
        }
    })
    
    return probj
