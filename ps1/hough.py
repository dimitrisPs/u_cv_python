"""Implementation of hough transformation functions."""


def hough_peaks(H, threshold=20, peakArea=1):
    """
    Find peaks in a Hough accumulator array.

    Parameters
    ----------
    H : np.array
        Hough accumulator array to search for peaks.
    Threshold : {20, int}, optional
        The threshold value to determine if a cell in the Hough accumulator
        represents a line. The default value is 20.
    peakArea : {1, int}, optional
        The number of neighbor pixel to consider when compute peaks. The pixel
        connectivity is according the moore neighborhood. Default value is 0.

    Returns
    -------
    A list with the location of peaks in Hough accumulator. in terms of d and
    theta and their corresponting pixel locations in H.

    """

def hough_lines_draw(src, hough_peaks):
    """
    Draw lines found in an image using Hough transform.

    Parameters
    ----------
    src : np.array
        The image to draw the lines.
    hough_peaks : list of tuples
        The locations of hough peaks in hough accumulator in terms of d and
        theta.

    Returns
    -------
    An image with the lines drawn.

    """

def hough_lines_acc(src, step_buckets=None, step_d=None):
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

def hough_circles_acc(src, r):
    """
    Computes the Hough transformation for circles in the input image.

    Parameters
    ----------
    src : np.array
        The image we want to compute the transformation of.
    r : The radius of the circle we need to find.

    Returns
    -------
    np.array
        The hough accumulator for circles.
    """


def find_circles():
    """
    Find circles in given radius range using Hough transform.

    Parameters
    ----------
    src : np.array
        The edge image needed to compute the transformation.
    rs : list
        A list with radiuses we need to check.

    Returns
    -------
    List with locations and radiuses of circles in the image.

    """
