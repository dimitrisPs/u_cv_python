# pso.py
"""Problem set 0 solutions."""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def swap_channels(src, ch_id1, ch_id2):
    """
    Swap the values of color channels in the input image.

    Parameters
    ----------
    src: np.array
        The input image.
    ch1: int [0-2]
        The number of the first channel. For bgr, the ids are:
            ch_id1 = 0 : blue channel.
            ch_id1 = 1 : green channel.
            ch_id1 = 2 : red channel.
    ch2: int [0-2]
        The number of the second channel. For bgr, the ids are:
            ch_id2 = 0 : blue channel.
            ch_id2 = 1 : green channel.
            ch_id2 = 2 : red channel.

    Returns
    -------
    np.array
        An image with swapped channels

    """
    out = src.copy()
    out[:, :, ch_id1] = src[:, :, ch_id2]
    out[:, :, ch_id2] = src[:, :, ch_id1]
    return out


def extract_channel(src, ch_id):
    """
    Extract a single channel of the src image according to the ch_idself.

    Parameters
    ----------
    Inputs:
        src: np.array
            the src image
        ch_id: int [0-1]
            The id of the channel you need to extract. For bgr, the ids are:
            ch_id = 0 : blue channel.
            ch_id = 1 : green channel.
            ch_id = 2 : red channel.

    Returns
    -------
        An np.array, containing a single channel according to ch_id.

    """
    return src[:, :, ch_id].copy()


def center_patch_loc(src, patch_height, patch_width):
    """
    Locate and return the start location of patch in the center of src.

    Parameters
    ----------
    src: np.array
        The src image.
    patch_width: int
        The width of the patch.
    patch_height: int
        The height of the patch.

    Returns
    -------
    a list containing the start row and column of the patch

    """
    row, col = src.shape
    center_row = row/2+1
    center_col = col/2+1
    patch_row_min = int(center_row - patch_height/2)
    patch_col_min = int(center_col - patch_width/2)

    return [patch_row_min, patch_col_min]


def add_noise_gaussian_mono(src, sigma=10):
    """
    Add gaussian noise to one channel image.

    Parameters
    ----------
    src : np.array
        The image to introduce noise.
    sigma : float
        The standard deviation of the gaussian noise.

    Returns
    -------
    np.array
        The original image with gaussian noise.

    """
    rows, cols = src.shape
    # Create a random image with the parameters of the gaussian destribution
    # needed.
    mean = 0
    noise = np.random.normal(mean, sigma, (rows, cols))
    out = np.int16(src) + np.int16(noise)
    # Add noise to src image.
    # The folowing 6 lines is for testing the statistics of the noise.
    # print(np.mean(mean))
    # print(np.std(noise))
    # print(np.min(noise))
    # print(np.max(noise))
    # plt.hist(noise.ravel(), 256, [-125, 125])
    # plt.show()
    return np.uint8(out)


def problem2():
    """Solution to the second part of ps0."""
    src1 = cv2.imread('./input/ps0-1-a-1.png')
    swapped = swap_channels(src1, 0, 2)
    img1_green = extract_channel(src1, 1)
    img1_red = extract_channel(src1, 0)
    # In case you need to display the results, uncommend the next 6 lines.
    # cv2.imshow('src', src1)
    # cv2.imshow('swapped', swapped)
    # cv2.imshow('green channel', img1_green)
    # cv2.imshow('red channel', img1_red)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('./output/ps0-2-a-1.png', swapped)
    cv2.imwrite('./output/ps0-2-b-1.png', img1_green)
    cv2.imwrite('./output/ps0-2-c-1.png', img1_red)


def problem3():
    """Solution to third part of ps0."""
    # Read input images.
    src1 = cv2.imread('./input/ps0-1-a-1.png')
    src2 = cv2.imread('./input/ps0-1-a-2.png')
    # extract green channels and consive them as gray channels
    gray1 = extract_channel(src1, 1)
    gray2 = extract_channel(src2, 1)
    # Find the start location of the patches in each image.
    patch_width = patch_height = 100
    patch1_s = center_patch_loc(gray1, patch_height, patch_width)
    patch2_s = center_patch_loc(gray2, patch_height, patch_width)
    # Make pointers to the patches in each image.
    patch1 = src1[patch1_s[0]:patch1_s[0]+patch_height,
                  patch1_s[1]:patch1_s[1]+patch_width]
    patch2 = src2[patch2_s[0]:patch2_s[0]+patch_height,
                  patch2_s[1]:patch2_s[1]+patch_width]
    # Copy the contents of the second patch to the first. This way we modify
    # the original image
    patch1[:, :] = patch2
    cv2.imwrite('./output/ps0-3-a-1.png', src1)


def problem4():
    """Solution to 4th part of ps0."""
    # Read the image file.
    src1 = cv2.imread('./input/ps0-1-a-1.png')
    # Extract the green channel and consive it as grayscale image of original.
    img1_green = extract_channel(src1, 2)
    # Use numpy functions to extract max, min, mean, standard deviation of
    # green channel and print them.
    print('max pixel value is ' + str(np.max(img1_green)) +
          '\nmin pixel value is ' + str(np.min(img1_green)) + '\nmean is ' +
          str(np.mean(img1_green)) + '\n standard deviation is ' +
          str(np.std(img1_green)))
    # Create a matrix with every element the mean value of green channel. The
    # size of this matrix will be the same as the green channel.
    mean_mat = np.full(img1_green.shape, np.uint8(np.mean(img1_green)))
    # Substruct mean_matrix from the grayscale image(green channel).
    result = img1_green - mean_mat
    std = np.std(img1_green)
    # Devide every element of the result matrix with the standard deviation of
    # green channel.
    for index, value in np.ndenumerate(result):
        result[index] = np.uint8(value/std)
    # Multiply the result matrix with the integer 10.
    result *= 10
    # Add mean matrix to the result matrix.
    result += mean_mat
    # Save the resulting image to a file.
    cv2.imwrite('./output/ps0-4-b-1.png', result)
    # Create a matrix to perform the 2 pixel shift. The matrix represents a 2D
    # affine transformation.
    m = np.array([[1., 0., -2.], [0., 1., 0.]])
    rows, cols = img1_green.shape
    # Translate the image according to the transformation matrix m.
    shift_img = cv2.warpAffine(img1_green, m, (cols, rows))
    # Save the resulting image to a file.
    cv2.imwrite('./output/ps0-4-c-1.png', shift_img)
    # Substruct The shfted grayscale image from the original grayscale image.
    diff_shift_img = img1_green - shift_img
    # Save the resulting image to a file.
    cv2.imwrite('./output/ps0-4-d-1.png', diff_shift_img)
    # For display perposes, uncoment the next 5 lines.
    # cv2.imshow('result', result)
    # cv2.imshow('shifted', shift_img)
    # cv2.imshow('diff', diff_shift_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def problem5():
    """Solution to the 5th part of ps0."""
    # Read the imput image.
    src = cv2.imread('./input/ps0-1-a-1.png')
    # Add gaussian noise to the original image.
    # Create noise channels
    mean = np.array([0, 0, 0])
    sigma = np.array([5, 5, 5])
    noise = np.zeros(src.shape, dtype=np.int8)
    cv2.randn(noise, mean, sigma)
    # Copy the src image to create the noise images.
    g_noise_green = src.copy()
    g_noise_blue = src.copy()
    # Add the noise to blue and green channel of output images. The images are
    # color encoded as bgr. Blue channel id is 0 and green channel is 1.
    g_noise_green[:, :, 1] = np.uint8(g_noise_green[:, :, 1] + noise[:, :, 1])
    g_noise_blue[:, :, 0] = np.uint8(g_noise_blue[:, :, 0] + noise[:, :, 0])
    # Uncoment the next 4 line to display the ouput images.
    # cv2.imshow('noise green', g_noise_green)
    # cv2.imshow('noise blue', g_noise_blue)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('./output/ps0-5-a-1.png', g_noise_green)
    cv2.imwrite('./output/ps0-5-b-1.png', g_noise_blue)


if __name__ == '__main__':
    # problem2()
    # problem3()
    # problem4()
    problem5()
