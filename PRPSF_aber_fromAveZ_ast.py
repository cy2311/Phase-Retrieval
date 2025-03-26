def PRPSF_aber_fromAveZ_ast(ims_Zcal_ave_plane1, index_record_Zplanes, empupil, setup):

    print('Generate pupil function and estimate aberration based on Phase retrieved method')

    print("[DEBUG] index_record_Zplanes:", index_record_Zplanes)
    print("[DEBUG] index_record_Zplanes shape:", index_record_Zplanes.shape)
    print("[DEBUG] index_record_Zplanes non-zero indices:", np.where(index_record_Zplanes)[0])
    
    probj = create_prpsf_dict()
    
    probj['PRstruct']['NA'] = empupil['NA']
    probj['PRstruct']['Lambda'] = empupil['Lambda']
    probj['PRstruct']['RefractiveIndex'] = empupil['nMed']
    probj['Pixelsize'] = empupil['Pixelsize']
    probj['PSFsize'] = 128
    probj['SubroiSize'] = empupil['imsz']
    probj['OTFratioSize'] = 60
    probj['ZernikeorderN'] = empupil['ZernikeorderN']
    probj['Enableunwrap'] = False
    probj['IterationNum'] = 25
    probj['IterationNumK'] = 5
    probj['CCDoffset'] = 0
    probj['Gain'] = 1
    
    print('Prepare PR data...')
    Zpos_plane1 = empupil['Z_pos'] + empupil['zshift']
    valid_idx = np.where(index_record_Zplanes)[0] 
    Zpos_plane1_sel = Zpos_plane1[valid_idx]
    
    probj['Zindstart'] = 1
    probj['Zindend'] = len(Zpos_plane1_sel)
    probj['Zindstep'] = 1

    probj['Zpos'] = Zpos_plane1_sel.astype(np.complex128)

    print("[DEBUG] probj.Zpos:", probj['Zpos'])
    print("[DEBUG] probj.Zpos shape:", probj['Zpos'].shape)

    probj['Xpos'] = np.zeros(len(Zpos_plane1_sel))
    probj['Ypos'] = np.zeros(len(Zpos_plane1_sel))
    probj['BeadData'] = np.transpose(ims_Zcal_ave_plane1[:, :, valid_idx], (1, 0, 2))

    prepdata(probj)
    precomputeParam(probj)

    print('First time PR...')
    genMpsf(probj)
    phaseretrieve(probj)
    genZKresult(probj)
    findOTFparam(probj)
    
    if not setup['is_imgsz']:
        findOTFparam_fixedsigma(probj, empupil['blur_sigma'])

    print(f'Shift mode in EMpupil is: {empupil["Zshift_mode"]}')

    # XYZ drift correciton
    if empupil['Zshift_mode']:
        for ii in range(1, 4):  
            C4 = probj['PRstruct']['Zernike_phase'][3].astype(np.complex128)

            def cost_function(x):
                return fitdefocus(probj, x, C4)  # x[0]=实部，x[1]=虚部

            initial_guess = [0.2, 0.0]  # [实部初始值, 虚部初始值]
            
            res = minimize(cost_function,
                         x0=initial_guess,
                         method='Nelder-Mead',
                         options={'xatol': 1e-4, 'fatol': 1e-4})
            
            zshift_real = res.x[0]  # 实部偏移量
            zshift_imag = res.x[1]  # 虚部偏移量（物理上应接近0）
            zshift_tmp = zshift_real  # 仅使用实部偏移量（MATLAB原逻辑）

            print(f"[DEBUG] 迭代 {ii} - 优化结果: real={zshift_real:.4f}, imag={zshift_imag:.4f}")

            empupil['zshift'] += zshift_tmp

            CXY = probj['PRstruct']['Zernike_phase'][1:3].astype(np.complex128)
            xyshift = (CXY * probj['PRstruct']['Lambda'] / 
                     (2 * np.pi * probj['Pixelsize'] * probj['PRstruct']['NA']))
            
            if 'BeadData' not in probj or probj['BeadData'] is None:
                raise ValueError("BeadData 未正确初始化!")
            
            tmp_input = np.zeros_like(probj['BeadData'])

            for i in range(probj['BeadData'].shape[2]):
                tmp_input[:, :, i] = FourierShift2D(
                    probj['BeadData'][:, :, i], 
                    delta=[-xyshift[1].real, -xyshift[0].real] 
                )
            
            probj['Zpos'] = (probj['Zpos'] + zshift_tmp).astype(np.complex128)
            probj['BeadData'] = tmp_input.copy()

            prepdata(probj)
            
            genMpsf(probj)
            phaseretrieve(probj)
            genZKresult(probj)
            findOTFparam(probj)
        
            if not setup['is_imgsz']:
                findOTFparam_fixedsigma(probj, empupil['blur_sigma'])

    if empupil['Zshift_mode'] == 2:
        empupil['zshift'] = 0  # 重置偏移量

    print(f'最终Z偏移量: {empupil["zshift"]:.4f} μm')
    print('更新后的Zernike系数(Z5-Z9):')
    zernike_coeffs = probj['PRstruct']['Zernike_phase'][4:9].astype(np.complex128)
    for i, coeff in enumerate(zernike_coeffs, start=5):
        print(f"  Z{i}: {coeff.real:.4f} + {coeff.imag:.4f}j")

    return probj, empupil
