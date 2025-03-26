def FourierShift2D(x, delta):
    """
    Shifts x by delta cyclically using the Fourier shift theorem.

    Parameters:
    x (numpy.ndarray): Input 2D array.
    delta (list or tuple): Shift values [delta_x, delta_y].

    Returns:
    y (numpy.ndarray): Shifted 2D array.
    """
    # The size of the matrix
    N, M = x.shape

    # FFT of the input signal
    X = np.fft.fft2(x)

    # Compute the shift in the frequency domain
    x_shift = np.exp(-1j * 2 * np.pi * delta[0] * np.hstack((np.arange(0, N//2), np.arange(-N//2, 0))) / N)
    y_shift = np.exp(-1j * 2 * np.pi * delta[1] * np.hstack((np.arange(0, M//2), np.arange(-M//2, 0))) / M)

    # Force conjugate symmetry for even-length signals
    if N % 2 == 0:
        x_shift[N//2] = np.real(x_shift[N//2])
    if M % 2 == 0:
        y_shift[M//2] = np.real(y_shift[M//2])

    # Apply the shift in the frequency domain
    Y = X * np.outer(x_shift, y_shift)

    # Inverse FFT to get the shifted signal
    y = np.fft.ifft2(Y)

    # Ensure the output is real if the input is real
    if np.isrealobj(x):
        y = np.real(y)

    return y
