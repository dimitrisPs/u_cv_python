"""Implementation of hough transformation functions."""
import cv2
import numpy as np


def are_neighbors(p1, p2, dim):
    """
    Check if two pixels are neighbors.

    Parameters
    ----------
    p1 : tuple
        Location of the first pixel.
    p2 : tuple
        Location of the second pixel.
    dim : tuple
        The dimention of the array the pixels are taken fromself.

    Returns
    -------
    boolean
        True if the pixels are considered neighbors, False otherwise.

    """
    # Compute the difference of the pixels in each dimention.
    row_dif = p1[0] - p2[0]
    col_dif = p1[1] - p2[1]
    # Compute their euclidean distance.
    diff = np.sqrt(row_dif**2 + col_dif**2)
    # Compute the maximum distance between two pixels, without taking into
    # consideration theta wrap.
    diag = np.sqrt(dim[0]**2 + dim[1]**2)
    # If the distance between the two pixels if grater than 0.05 of the max
    # possible distance, consider them neighbors.
    if diag-diff < 0.95*diag:
        return False
    return True


def duplicate_removal(peaks, dim):
    """
    Remove Accumulator peaks that are considered neighbors.

    Parameters
    ----------
    peaks : List of tuples
        Each tuple has the following format: (d, theta, row, column). d is in
        pixels, theta in rads, and row, column is the location of the peak in
        Hough accumulator Array.
    dim : tuple
        The dimentions of Hough accumulator.

    Returns
    -------
    List of tuples.
        peaks list, without most of the duplicates.

    """
    lines2 = []
    flag = False
    # For every peaks i
    for i in range(len(peaks)):
        # Check all the fllowing peaks
        for j in range(len(peaks[i+1:])):
            peak1 = peaks[i][1]
            peak2 = peaks[i+j+1][1]
            if are_neighbors(peak1, peak2, dim):
                flag = True
                break
        # If the peak i has at least one following neighbor, it is a duplicate.
        if flag is True:
            flag = False
            continue
        # If the peak i dose not have any neighbor, then store it in the List
        # with the non duplicates.
        lines2.append(peaks[i])
    return lines2


def sum_neighbors(H, index, dist):
    """
    Sum the cell values of all the cells in the neighborhood of index cell.

    Parameters
    ----------
    H : np.array
        The hough accumulator array.
    index : tuple
        Contains the row and column of the central cell.
    dist :  int
        The distance of the farthest neighbor cell we want to include.

    Returns
    -------
    int
        The sum of values of all the cells in the neighborhood

    """
    # Initialize the sum variable s to zero.
    s = 0
    # Extract the rows and columns of H.
    rows, cols = H.shape
    # For every pixel in the neighborhood of pixel with index location, sum
    # their values.
    for i in range(-dist, dist+1, 1):
        for j in range(-dist, dist+1, 1):
            row = j+index[0]
            col = i+index[1]
            # Wrap around the matrix.
            if row >= rows:
                row = row - rows
            if col >= cols:
                col = col - cols
            s += H[row, col]
    return s


def hough_peaks(H, src_dim, threshold=150, peakArea=0):
    """
    Find peaks in a Hough accumulator array.

    Parameters
    ----------
    H : np.array
        Hough accumulator array to search for peaks.
    src_dim : tuple
        The size of source image array.
    Threshold : {20, int}, optional
        The threshold value to determine if a cell in the Hough accumulator
        represents a line. The default value is 150.
    dist : {1, int}, optional
        The distance of the farthest neighbor cell we want to include. The
        pixel connectivity is according the moore neighborhood. Default value
        is 0.

    Returns
    -------
    A list of tuples
        Each tuple has the following format: (d, theta, row, column). d is in
        pixels, theta in rads, and row, column is the location of the peak in
        Hough accumulator Array.

    """
    # Extract the size of the H matrix.
    rows, cols = H.shape
    # Compute the diagonal of the original image. This values is needed to
    # compute d later.
    r, c = src_dim
    diagonal = np.sqrt(r**2 + c**2)
    # Initialize an empty array to store the local maximums of H.
    points = []
    # For every pixel of H
    for index, pixel in np.ndenumerate(H):
        # If the pixel is less than 0.8 of threshold, continue
        if pixel < threshold:
            continue
        # Sum the values of all the pixels with distande peakArea
        s = sum_neighbors(H, index, peakArea)
        # Normalize the sum. The normalization is done by deviding the size of
        # neighbor in one dimention^(3/2) and not in boath. This is done
        # because not all the neighbor pixel have high values.
        s /= (2*peakArea+1)**(3/2)
        # Check if index pixel is peak.
        if s > threshold:
            # Compute theta and d
            theta = index[1] / float(cols) * np.pi * 2
            d = index[0]/float(rows) * diagonal
            points.append(((d, theta), index))
    # Remove duplicate lines. At this point, there may be peaks in Hough
    # accumulator very close together and they are represent the same line.
    duplicate_removal(points, H.shape)
    return points


def hough_lines_draw(src, hough_peaks):
    """
    Draw lines found in an image using Hough transform.

    Parameters
    ----------
    src : np.array [3ch]
        The image to draw the lines.
    hough_peaks : list of tuples
        Contains tuples of the format (d, theta, row, column), one for every
        peak. d is the distance in pixels, theta the angle and the two values
        together represent one line. The following two parameters are the
        location of the peak in Hough accumulator array.

    Returns
    -------
    An image with the lines drawn in red color.

    """
    out = src.copy()
    # for every peak in the input List
    for val in hough_peaks:
        # Extract d, theta.
        d, theta = val[0]
        # Compute the point where the vector d and the line intersect
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * d
        y0 = b * d
        # Create two collinear points that lie on the d, theta line and are far
        # away from the point (x0, y0)
        x1 = np.int(x0 + 1000*(-b))
        y1 = np.int(y0 + 1000*(a))
        x2 = np.int(x0 - 1000*(-b))
        y2 = np.int(y0 - 1000*(a))
        p1 = (x1, y1)
        p2 = (x2, y2)
        # Draw the line between points p1 and p2.
        out = cv2.line(out, p1, p2, (0, 0, 255), thickness=2)
    return out


def hough_lines_acc(src, step_theta=None, step_d=None):
    """
    Compute the Hough transformation for lines in the input image.

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
                theta = theta_cell / float(cols) * np.pi * 2
                # Compute d.
                d = x*np.cos(theta) + y*np.sin(theta)
                # Map d to an index in accumulator matrix.
                d = np.int((d/diagonal)*(rows))
                if d < 0:
                    continue
                # Increse the corresponting bucket.
                H[d, theta_cell] += 1
    # Normalize Hough transform.
    H *= (255./np.max(H))
    H = np.uint8(H)
    return H


def enchance_acc(H):
    """
    Enchance accumulator matrix, using histogram equalization.

    This must be used only for visual presentation perposes. The histogram of
    Hough accoumulator is equalized using CLAHE method.

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


# if __name__ == '__main__':
