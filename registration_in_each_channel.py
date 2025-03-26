import numpy as np

def registration_in_each_channel(im1, im2):
    """
    Calculate the shift between two images using cross-correlation and DFT registration.

    Parameters:
    im1 (numpy.ndarray): Reference image.
    im2 (numpy.ndarray): Image to align with the reference.

    Returns:
    shift1 (float): Shift in the first dimension (rows).
    shift2 (float): Shift in the second dimension (columns).
    similarity (float): Similarity score between the aligned images.
    """
    # Perform DFT registration to find the shift
    output, _ = dftregistration(np.fft.fft2(im1), np.fft.fft2(im2), usfac=10)
    shift1 = output[2]
    shift2 = output[3]

    # Apply the calculated shift to the second image
    im2_shift = FourierShift2D(im2, [shift1, shift2])

    # Calculate the similarity between the aligned images (excluding borders)
    similarity, _ = cc2(im1[2:-2, 2:-2], im2_shift[2:-2, 2:-2])

    return shift1, shift2, similarity
