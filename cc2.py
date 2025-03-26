import numpy as np

def cc2(ref, img):
    """
    Calculate the 2D cross-correlation between two images.

    Parameters:
    ref (numpy.ndarray): Reference image.
    img (numpy.ndarray): Image to compare with the reference.

    Returns:
    maxa (float): Maximum normalized cross-correlation value.
    cc (numpy.ndarray): Cross-correlation matrix.
    """
    # Normalize the reference image
    ref = ref - np.mean(ref)
    ref = ref / np.std(ref)

    # Normalize the input image
    img = img - np.mean(img)
    img = img / np.std(img)

    # Compute the cross-correlation using FFT
    cc = np.abs(np.fft.ifft2(np.fft.fft2(ref) * np.conj(np.fft.fft2(img))))

    # Compute the maximum normalized cross-correlation value
    maxa = (1 / img.size) * np.max(cc)

    return maxa, cc
