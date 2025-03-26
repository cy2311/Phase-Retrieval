def gen_aberPSF_fromPR_ast(probj, empupil):

    imsz = empupil['imsz']
    Numimage = len(empupil['Z_pos'])
    ref_plane1 = np.zeros((imsz, imsz, Numimage), dtype=np.complex128)

    label = probj['PRstruct']['Zernike_phase'][4:25]  # 提取 Zernike 系数

    for ii in range(Numimage):

        Ztrue_plane1 = empupil['Z_pos'][ii] + empupil['zshift']
        Ztrue_plane1 = np.round(Ztrue_plane1, 2)
        print(f"Frame {ii + 1}/{Numimage}: Ztrue_plane1 = {Ztrue_plane1}")


        PRstruct = probj['PRstruct']
        PRstruct['SigmaX'] = empupil['blur_sigma']
        PRstruct['SigmaY'] = empupil['blur_sigma']

        if abs(label[0] - empupil['init_z'][0]) > 0.5:

            PRstruct['Pupil'] = {
                'phase': np.zeros((128, 128)),
                'mag': np.zeros((128, 128))
            }
            phaseZ = np.zeros(25)
            phaseZ[4:25] = label
            phaseZ[4] = empupil['init_z'][0]
            magZ = np.zeros(25)
            magZ[0] = 1
            PRstruct['Zernike_phase'] = phaseZ
            PRstruct['Zernike_mag'] = magZ


        psfobj = PSF_zernike(PRstruct)
        psfobj.Xpos = np.array([0])
        psfobj.Ypos = np.array([0])
        psfobj.Zpos = np.array([Ztrue_plane1])
        psfobj.Boxsize = imsz
        psfobj.Pixelsize = probj['Pixelsize']
        psfobj.PSFsize = 128
        psfobj.nMed = PRstruct['RefractiveIndex']
        psfobj.precomputeParam()


        if abs(label[0] - empupil['init_z'][0]) > 0.5:
            psfobj._compute_zernike_basis()
            psfobj.genPupil()
            psfobj.genPSF_2()
        else:
            psfobj.setPupil()
            psfobj.genPSF_2()

        psfobj.scalePSF('normal')

        I = np.tile(5000, (imsz, imsz, 1))
        psf = psfobj.ScaledPSFs
        normf = np.sum(psfobj.Pupil['mag']**2)
        img = (psf / normf) * I
        bg = 0
        ref_plane1[:, :, ii] = np.sum(img, axis=2) + bg

    return ref_plane1
