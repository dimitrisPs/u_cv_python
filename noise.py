"""Functions to add noise to images."""

import numpy as np


def add_noise_sp(src, percentage=10, ratio=0.5):
    """
    Apply salt and pepper noise to src image.

    Parameters
    ----------
    inputs:
        src: np.array
            The src image.
        percentage: float [0-100]
            The percentage of the pixels we want to add noise.
        ratio: float [0-1]
            The ration of the black vs white pixels
    outputs:
        Returns a copy of src with added noise.

    """
    #
    out = src.copy()
    row, col, ch = out.shape
    percentage *= 0.01
    perc_salt = percentage * ratio
    perc_pepper = percentage * (1-ratio)
    # To add noise, select len_salt and len_pepper pixels randomly and set
    # their values to 0 and 255 respectively.
    # Generate a mask array, which indicate which pixel will be salt,
    # pepper, or will have no noise.
    noise_mask = np.random.choice(['salt', 'pepper', 'orig'],
                                  (row, col),
                                  p=[perc_salt, perc_pepper, (1-percentage)])
    # Iterate through all the cells of the mask array, and add noise to the
    # output array according to noise_mask values
    for index, value in np.ndenumerate(noise_mask):
        if value == 'salt':
            out[index] = [255, 255, 255]
        elif value == 'pepper':
            out[index] = [0, 0, 0]
    return out


def add_noise_gaussian(src, var=1):
    """
    Apply gaussian noise to sourse image.

    Parameters
    ----------
    inputs:
        src: np.array
            The src image
        var: float
            The variance of the distribution. The higher the value, the greater
            the noise.
    outputs:
        Returns the src image with added noise.

    """
    # Create the noise matrix Acconrding to the src imageself.
    # Extract the size and number of channels from source.
    row, col, ch = src.shape
    # The noise will be additive. The mean will be 0.
    mean = 0
    sigma = var ** .5  # Standard deviation.
    # Create noise matrix.
    noise = np.random.normal(mean, sigma, (row, col, ch))
    # Cast the noise matrix to uint8 before adding it to the original image.
    # The original image is np.uint8 and the noise matrix is np.float64.
    # Without casting the noise matrix, the output will be np.float64 and the
    # the the output will be a gray image.
    noise = np.uint8(noise)
    # Add the src image and the noise and return.
    return noise
