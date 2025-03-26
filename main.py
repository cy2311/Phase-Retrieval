import numpy as np
from scipy.ndimage import gaussian_filter
import scipy
from multiprocessing import Pool
from functools import partial

class INSPRModelGeneratorAst:
    def __init__(self, subregion_ch1, setup, pupil_para):
        self.subregion_ch1 = subregion_ch1
        self.setup = setup
        self.pupil_para = pupil_para
        self.pupil_stop = False

    def generate_model(self):
        empupil = self._initialize_empupil()
        probj = gen_initPupil(empupil)

        for iter in range(empupil['iter']):
            print(f'Iteration: {iter + 1} ...')

            ref_plane1 = gen_aberPSF_fromPR_ast(probj, empupil)

            ims_Zcal_ave_plane1, index_record_Zplanes = classify_onePlane_par(self.subregion_ch1, ref_plane1, empupil)

            probj, empupil = PRPSF_aber_fromAveZ_ast(ims_Zcal_ave_plane1, index_record_Zplanes, empupil, self.setup)

        index_number = np.sum(index_record_Zplanes == 1)

        if index_number <= 5:
            print('Warning! The range of localization isn’t enough for reliable model generation!')

    def _initialize_empupil(self):
        empupil = {}
        empupil['NA'] = self.setup['NA']
        empupil['Lambda'] = self.setup['Lambda']
        empupil['nMed'] = self.setup['RefractiveIndex']
        empupil['Pixelsize'] = self.setup['Pixelsize']
        empupil['imsz'] = self.subregion_ch1.shape[0]
        empupil['PSFsize'] = empupil['imsz'] 
        empupil['Boxsize'] = empupil['imsz']  
        empupil['Z_pos'] = self.pupil_para['Z_pos']
        empupil['bin_lowerBound'] = self.pupil_para['bin_lowerBound']
        empupil['min_similarity'] = self.pupil_para['min_similarity']
        empupil['iter'] = self.pupil_para['iter']
        empupil['blur_sigma'] = self.pupil_para['blur_sigma']
        if self.setup['is_imgsz'] == 0:
            empupil['blur_sigma'] = 1
        empupil['init_z'] = self.pupil_para['init_z']
        empupil['ZernikeorderN'] = self.pupil_para['ZernikeorderN']
        empupil['Zernike_sz'] = (empupil['ZernikeorderN'] + 1) ** 2
        empupil['Zshift_mode'] = self.pupil_para['Zshift_mode']
        empupil['iter_mode'] = 0
        empupil['zshift'] = 0
        return empupil

    def _gen_init_pupil(self, empupil):
        # Placeholder for initial pupil generation
        probj = {}
        return probj

setup = {
    'NA': 1.35,
    'Lambda': 0.68,
    'RefractiveIndex': 1.406,
    'Pixelsize': 0.12,
    'is_imgsz': 1
}

pupil_para = {
    'Z_pos': np.round(np.arange(-0.5, 0.55, 0.05),2),
    'bin_lowerBound': 50,
    'min_similarity': 0.7,
    'iter': 8,
    'blur_sigma': 1.5,
    'init_z': np.round(np.arange(-0.5, 0.55, 0.05),2),
    'ZernikeorderN': 7,
    'Zshift_mode': 0
}

#subregion_ch1 = np.random.rand(64, 64)
#r"C:\Users\Siyuan\Desktop\Phase_retrive\Phase_retrive_py\data\subregions.npy"
#subregion_ch1 = scipy.io.loadmat(r"C:\Users\Siyuan\Desktop\Phase_retrive\Phase_retrive_py\data\rawData.mat")

subregion_ch1 = np.load(r"C:\Users\Siyuan\Desktop\Phase_retrive\Phase_retrive_py\data\subregions.npy")
subregion_ch1 = np.transpose(subregion_ch1, (1, 2, 0))

print(subregion_ch1.shape)

model_generator = INSPRModelGeneratorAst(subregion_ch1, setup, pupil_para)
model_generator.generate_model()
