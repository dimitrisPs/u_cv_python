"""Implementation of hough transformation functions."""
import cv2
import numpy as np

# def hough_peaks(H, threshold=20, peakArea=1):
#     """
#     Find peaks in a Hough accumulator array.
#
#     Parameters
#     ----------
#     H : np.array
#         Hough accumulator array to search for peaks.
#     Threshold : {20, int}, optional
#         The threshold value to determine if a cell in the Hough accumulator
#         represents a line. The default value is 20.
#     peakArea : {1, int}, optional
#         The number of neighbor pixel to consider when compute peaks. The pixel
#         connectivity is according the moore neighborhood. Default value is 0.
#
#     Returns
#     -------
#     A list with the location of peaks in Hough accumulator. in terms of d and
#     theta and their corresponting pixel locations in H.
#
#     """

# def hough_lines_draw(src, hough_peaks):
#     """
#     Draw lines found in an image using Hough transform.
#
#     Parameters
#     ----------
#     src : np.array
#         The image to draw the lines.
#     hough_peaks : list of tuples
#         The locations of hough peaks in hough accumulator in terms of d and
#         theta.
#
#     Returns
#     -------
#     An image with the lines drawn.
#
#     """

def hough_lines_acc(src, step_theta=None, step_d=None):
    """
    Computes the Hough transformation for lines in the input image.

    Parameters
    ----------
    src : np.array
        The image we want to compute the transformation of.
    step_buckets : {None, int}, optional
        The Number of buckets in theta dimention. Default value is None and it
        will create as many thera cells as the columns of src image.
    step_d : {None, int}, optional
        The number of buckets in d dimention. Default value is None and it will
        create as many d cells as the rows of src image.

    Returns
    -------
    np.array
        The hough accumulator for lines.
    """
    if step_theta is None:
        cols = src.shape[1]
    else:
        cols = step_theta
    if step_d is None:
        rows = src.shape[0]
    else:
        rows = step_d
    # Create an empty array, the size of src.
    H = np.zeros((rows, cols), dtype=float)
    # Compute the maximum r. diagonal is the max d of any line in the image
    r, c = src.shape
    diagonal = np.sqrt(r**2 + c**2)
    # For every edge pixel in src, compute it's hough transform for lines and
    # update H.
    for index, pixel in np.ndenumerate(src):
        # Edge pixels have value 255.
        if pixel == 255:
            # For every edge pixel there is a corresponting sinusoid. Compute
            # The buckets that this sinusoid occupies.
            x = index[1]
            y = index[0]
            for theta_cell in range(cols):
                # For every possible thera, compute d for point x,y.
                # Convert array indexes to polar angles.
                theta = theta_cell / float(cols) * np.pi
                # Compute d.
                d = x*np.cos(theta) + y*np.sin(theta)
                # Map d to an index in accumulator matrix.
                d = np.int((d/diagonal)*(rows))
                # Increse the corresponting bucket.
                H[d, theta_cell] += 1
    # Normalize Hough transform.
    H *= (255./np.max(H))
    H = np.uint8(H)
    return H


def enchance_acc(H):
    """
    Enchance accumulator matrix, using histogram equalization.

    This must be used only for visual representation perpose. The histogram is
    equalized using CLAHE method.

    Parameters
    ----------
    H : np.array
        Accumulator matrix from Houng transformation.

    Returns
    -------
    np.array
        Enchanced accumulator matrix.

    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(H)
    return np.uint8(cl1)



# def hough_circles_acc(src, r):
#     """
#     Computes the Hough transformation for circles in the input image.
#
#     Parameters
#     ----------
#     src : np.array
#         The image we want to compute the transformation of.
#     r : The radius of the circle we need to find.
#
#     Returns
#     -------
#     np.array
#         The hough accumulator for circles.
#     """
#
#
# def find_circles():
#     """
#     Find circles in given radius range using Hough transform.
#
#     Parameters
#     ----------
#     src : np.array
#         The edge image needed to compute the transformation.
#     rs : list
#         A list with radiuses we need to check.
#
#     Returns
#     -------
#     List with locations and radiuses of circles in the image.
#
#     """


if __name__ == '__main__':
    # For testing.
    src = cv2.imread('./input/1.png')
    edge_img = cv2.Canny(src, 100, 200)
    H = hough_lines_acc(edge_img,100,100)
    cv2.imshow('hough', H)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
