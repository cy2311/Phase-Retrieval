import numpy as np

def dftregistration(buf1ft, buf2ft, usfac=1):
    """
    Efficient subpixel image registration by cross-correlation.

    Parameters:
    buf1ft (numpy.ndarray): Fourier transform of reference image (DC in (0,0)).
    buf2ft (numpy.ndarray): Fourier transform of image to register (DC in (0,0)).
    usfac (int): Upsampling factor (default = 1).

    Returns:
    output (list): [error, diffphase, row_shift, col_shift]
    Greg (numpy.ndarray): (Optional) Fourier transform of registered version of buf2ft.
    """
    nr, nc = buf2ft.shape
    Nr = np.fft.ifftshift(np.arange(-nr // 2, np.ceil(nr / 2)))
    Nc = np.fft.ifftshift(np.arange(-nc // 2, np.ceil(nc / 2)))

    if usfac == 0:
        # Simple computation of error and phase difference without registration
        CCmax = np.sum(buf1ft * np.conj(buf2ft))
        row_shift = 0
        col_shift = 0
    elif usfac == 1:
        # Single pixel registration
        CC = np.fft.ifft2(buf1ft * np.conj(buf2ft))
        CCabs = np.abs(CC)
        row_shift, col_shift = np.unravel_index(np.argmax(CCabs), CCabs.shape)
        CCmax = CC[row_shift, col_shift] * nr * nc
        # Now change shifts so that they represent relative shifts and not indices
        row_shift = Nr[row_shift]
        col_shift = Nc[col_shift]
    elif usfac > 1:
        # Start with usfac == 2
        CC = np.fft.ifft2(FTpad(buf1ft * np.conj(buf2ft), [2 * nr, 2 * nc]))
        CCabs = np.abs(CC)
        row_shift, col_shift = np.unravel_index(np.argmax(CCabs), CCabs.shape)
        CCmax = CC[row_shift, col_shift] * nr * nc
        # Now change shifts so that they represent relative shifts and not indices
        Nr2 = np.fft.ifftshift(np.arange(-nr, np.ceil(nr)))
        Nc2 = np.fft.ifftshift(np.arange(-nc, np.ceil(nc)))
        row_shift = Nr2[row_shift] / 2
        col_shift = Nc2[col_shift] / 2
        # If upsampling > 2, then refine estimate with matrix multiply DFT
        if usfac > 2:
            # DFT computation
            row_shift = np.round(row_shift * usfac) / usfac
            col_shift = np.round(col_shift * usfac) / usfac
            dftshift = int(np.ceil(usfac * 1.5) // 2)
            # Matrix multiply DFT around the current shift estimate
            CC = np.conj(dftups(buf2ft * np.conj(buf1ft), int(np.ceil(usfac * 1.5)), int(np.ceil(usfac * 1.5)), usfac,
                                dftshift - row_shift * usfac, dftshift - col_shift * usfac))
            # Locate maximum and map back to original pixel grid
            CCabs = np.abs(CC)
            rloc, cloc = np.unravel_index(np.argmax(CCabs), CCabs.shape)
            CCmax = CC[rloc, cloc]
            rloc = rloc - dftshift - 1
            cloc = cloc - dftshift - 1
            row_shift = row_shift + rloc / usfac
            col_shift = col_shift + cloc / usfac

        # If its only one row or column the shift along that dimension has no effect
        if nr == 1:
            row_shift = 0
        if nc == 1:
            col_shift = 0

    # Compute error and phase difference
    rg00 = np.sum(np.abs(buf1ft) ** 2)
    rf00 = np.sum(np.abs(buf2ft) ** 2)
    error = 1.0 - np.abs(CCmax) ** 2 / (rg00 * rf00)
    error = np.sqrt(np.abs(error))
    diffphase = np.angle(CCmax)

    output = [error, diffphase, row_shift, col_shift]

    # Compute registered version of buf2ft
    if usfac > 0:
        Nc, Nr = np.meshgrid(Nc, Nr)
        Greg = buf2ft * np.exp(1j * 2 * np.pi * (-row_shift * Nr / nr - col_shift * Nc / nc))
        Greg = Greg * np.exp(1j * diffphase)
        return output, Greg
    elif usfac == 0:
        Greg = buf2ft * np.exp(1j * diffphase)
        return output, Greg
    return output


def dftups(inp, nor, noc, usfac=1, roff=0, coff=0):
    """
    Upsampled DFT by matrix multiplies.

    Parameters:
    inp (numpy.ndarray): Input array.
    nor (int): Number of rows in output.
    noc (int): Number of columns in output.
    usfac (int): Upsampling factor (default = 1).
    roff (int): Row offset (default = 0).
    coff (int): Column offset (default = 0).

    Returns:
    out (numpy.ndarray): Upsampled DFT.
    """
    nr, nc = inp.shape
    # Compute kernels and obtain DFT by matrix products
    kernc = np.exp((-1j * 2 * np.pi / (nc * usfac)) * (np.fft.ifftshift(np.arange(nc)) - np.floor(nc / 2)).reshape(-1, 1) *
                   (np.arange(noc) - coff))
    kernr = np.exp((-1j * 2 * np.pi / (nr * usfac)) * (np.arange(nor) - roff).reshape(-1, 1) *
                   (np.fft.ifftshift(np.arange(nr)) - np.floor(nr / 2)))
    out = kernr @ inp @ kernc
    return out


def FTpad(imFT, outsize):
    """
    Pads or crops the Fourier transform to the desired output size.

    Parameters:
    imFT (numpy.ndarray): Input complex array with DC in (0,0).
    outsize (list): Output size of array [ny, nx].

    Returns:
    imFTout (numpy.ndarray): Output complex image with DC in (0,0).
    """
    if len(imFT.shape) != 2:
        raise ValueError("Input array must be 2D.")
    Nout = outsize
    Nin = imFT.shape
    imFT = np.fft.fftshift(imFT)
    center = np.floor(np.array(Nin) / 2).astype(int)

    imFTout = np.zeros(Nout, dtype=complex)
    centerout = np.floor(np.array(Nout) / 2).astype(int)

    # Calculate padding or cropping regions
    cenout_cen = centerout - center
    imFTout[max(cenout_cen[0], 0):min(cenout_cen[0] + Nin[0], Nout[0]),
            max(cenout_cen[1], 0):min(cenout_cen[1] + Nin[1], Nout[1])] = \
        imFT[max(-cenout_cen[0], 0):min(-cenout_cen[0] + Nout[0], Nin[0]),
             max(-cenout_cen[1], 0):min(-cenout_cen[1] + Nout[1], Nin[1])]

    imFTout = np.fft.ifftshift(imFTout) * Nout[0] * Nout[1] / (Nin[0] * Nin[1])
    return imFTout
