# pso.py
"""Problem set 0 solutions."""
import cv2



def swap_channels(src, ch_id1, ch_id2):
    """
    Swap the values of color channels in the input image.

    Parameters
    ----------
    src: np.array
        The input image.
    ch1: int [0-2]
        The number of the first channel.
            ch_id1 = 0 : red channel.
            ch_id1 = 1 : green channel.
            ch_id1 = 2 : blue channel.
    ch2: int [0-2]
        The number of the second channel.
            ch_id2 = 0 : red channel.
            ch_id2 = 1 : green channel.
            ch_id2 = 2 : blue channel.

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
            The id of the channel you need to extract.
            ch_id = 0 : red channel.
            ch_id = 1 : green channel.
            ch_id = 2 : blue channel.

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


if __name__ == '__main__':
    problem3()
