"""Implementation of problem set 1."""
import cv2
import hough
import numpy as np


def auto_canny(image, sigma=0.33):
    """
    Zero parameter edge extractor, using Canny algorithm.

    More information about this method can be found here:
    https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/

    Parameters
    ----------
    image : np.array
        Image to apply Canny edge detection algorithm.
    sigma : float
        standard deviation of the log kernel.

    Return
    ------
    edge : np.Array
        the edge image.

    """
    # Compute the median of the single channel pixel intensities
    v = np.median(image)
    # Compute the upper and lower thresholds using the computed median.
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # Compute the edge image.
    edge = cv2.Canny(image, lower, upper)
    # return the edged image
    return edge


def problem1():
    """Solution to part 1 of ps1."""
    # Load the input image.
    src = cv2.imread('./input/ps1-input0.png')
    # Compute the edge image using canny edge method.
    edge_img = cv2.Canny(src, 100, 200)
    # Save the resulting image.
    cv2.imwrite('./output/ps1-1-a-1.png', edge_img)
    # Uncoment the next 4 lines to display the resulting image.
    # cv2.imshow('original', src)
    # cv2.imshow('edge', edge_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def problem2():
    """Solution ot part 2 of ps1."""
    # Load the input image.
    src = cv2.imread('./input/ps1-input0.png')
    # Compute the edge image using canny edge method.
    edge_img = cv2.Canny(src, 100, 200)
    # Compute the Hough line transformation.
    H = hough.hough_lines_acc(edge_img)
    # Perform a histogram equalization to enchance the image.
    H_ench = hough.enchance_acc(H)
    # Save the result.
    cv2.imwrite('./output/ps1-2-a-1.png', H_ench)
    # Find peaks in accumulator array H.
    peaks = hough.hough_peaks(H, edge_img.shape, 150, 0)
    # Convert enchanced Hough image from gray to color to draw the peaks.
    H_ench = cv2.cvtColor(H_ench, cv2.COLOR_GRAY2RGB)
    # For each peak, draw a red dot in Hough accumulator array.
    H_peak = H_ench.copy()
    for param, pixel in peaks:
        cv2.circle(H_peak, (pixel[1], pixel[0]), 2, (0, 0, 255), -1)
    # Save the result.
    cv2.imwrite('./output/ps1-2-b-1.png', H_peak)
    # Draw the lines in the original image according to peaks List.
    line_img = hough.hough_lines_draw(src, peaks)
    # Save the result.
    cv2.imwrite('./output/ps1-2-c-1.png', line_img)
    # Uncoment the next 5 lines for display perposes.
    cv2.imshow('Hough Accumulator', H_ench)
    cv2.imshow('Hough Accumulator peaks', H_peak)
    cv2.imshow('line image', line_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def problem3():
    """Solution ot part 3 of ps1."""
    # Load the input image.
    src = cv2.imread('./input/ps1-input0-noise.png')
    # Remove noise with gaussian filter.
    src_smooth = cv2.GaussianBlur(src, (19, 19), 5)
    # Save smouthed image.
    cv2.imwrite('./output/ps1-3-a-1.png', src_smooth)
    # Compute the edge images using canny edge method.
    edge_img = cv2.Canny(src, 100, 200)
    edge_img_smoothed = cv2.Canny(src_smooth, 0, 50)
    # Save the original and smoothed images.
    cv2.imwrite('./output/ps1-3-b-1.png', edge_img)
    cv2.imwrite('./output/ps1-3-b-2.png', edge_img_smoothed)
    # Compute the Hough line transformation.
    H = hough.hough_lines_acc(edge_img_smoothed)
    # Perform a histogram equalization to enchance the image.
    H_ench = hough.enchance_acc(H)
    # Find peaks in accumulator array H.
    peaks = hough.hough_peaks(H, edge_img.shape, 140, 0)
    # Convert enchanced Hough image from gray to color to draw the peaks.
    H_ench = cv2.cvtColor(H_ench, cv2.COLOR_GRAY2RGB)
    # For each peak, draw a red dot in Hough accumulator array.
    H_peak = H_ench.copy()
    for param, pixel in peaks:
        cv2.circle(H_peak, (pixel[1], pixel[0]), 2, (0, 0, 255), -1)
    # Save the result.
    cv2.imwrite('./output/ps1-3-c-1.png', H_peak)
    # Draw the lines in the original image according to peaks List.
    line_img = hough.hough_lines_draw(src, peaks)
    # Save the result.
    cv2.imwrite('./output/ps1-3-c-2.png', line_img)
    # Uncoment the next 8 lines for display perposes.
    # cv2.imshow('smoothed original', src_smooth)
    # cv2.imshow('original edge image', edge_img)
    # cv2.imshow('smoothed edge image', edge_img_smoothed)
    # cv2.imshow('Hough Accumulator', H_ench)
    # cv2.imshow('Hough Accumulator peaks', H_peak)
    # cv2.imshow('line image', line_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def problem4():
    """Solution to the 4th part of ps1."""
    # Load image.
    src = cv2.imread('./input/ps1-input1.png')
    # Create a grayscale image from original.
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # smooth image to find edges.
    smoothed = cv2.GaussianBlur(src_gray, (5, 5), 8)
    # extract edges using canny edge algorithm.
    edge_img = auto_canny(smoothed, 5)
    H = hough.hough_lines_acc(edge_img)
    # Find peak points in Hough accumulator.
    peaks = hough.hough_peaks(H, edge_img.shape, 190)
    # perform histogram equalizaton to Hough accumulator to diplay it.
    H_ench = hough.enchance_acc(H)
    # Convert enchanced Hough image from gray to color to draw the peaks.
    H_ench = cv2.cvtColor(H_ench, cv2.COLOR_GRAY2RGB)
    src_gray = cv2.cvtColor(src_gray, cv2.COLOR_GRAY2RGB)
    # For each peak, draw a red dot in Hough accumulator array.
    H_peak = H_ench.copy()
    for param, pixel in peaks:
        cv2.circle(H_peak, (pixel[1], pixel[0]), 1, (0, 0, 255), -1)
    # Draw found lines in the original image.
    line_img = hough.hough_lines_draw(src_gray, peaks)
    # Save the result.
    cv2.imwrite('./output/ps1-4-a-1.png', smoothed)
    cv2.imwrite('./output/ps1-4-b-1.png', edge_img)
    cv2.imwrite('./output/ps1-4-c-1.png', H_peak)
    cv2.imwrite('./output/ps1-4-c-2.png', line_img)
    cv2.waitKey(0)


def problem5():
    """Solution to part 5 of ps1."""
    src = cv2.imread('./input/ps1-input1.png')
    # Convert src img from color to grayscale.
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Smoth image using gaussina blur.
    filtered = cv2.GaussianBlur(src_gray, (5, 5), 1.9)
    # Compute the edge image using Canny edge detection algorithm.
    edge_img = auto_canny(filtered, 3)
    # Define the radiuses to check.
    r = range(18, 30, 2)
    # Find circles in the image.
    peaks = hough.find_circles(edge_img, r)
    # Make copies of grayscale to draw the resaults later.
    out = src_gray.copy()
    out_20 = src_gray.copy()
    # Convert grayscale to color, in order to mark the circles in color.
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    out_20 = cv2.cvtColor(out_20, cv2.COLOR_GRAY2RGB)
    # Draw every detected circle in the grayscale src image.
    for radius, pixel in peaks:
        if radius == 20:
            cv2.circle(out_20, (pixel[1], pixel[0]), 2, (0, 255, 0), -1)
            cv2.circle(out_20, (pixel[1], pixel[0]), radius, (0, 0, 255), 2)
        cv2.circle(out, (pixel[1], pixel[0]), 2, (0, 255, 0), -1)
        cv2.circle(out, (pixel[1], pixel[0]), radius, (0, 0, 255), 2)
    # Save results.
    cv2.imwrite('./output/ps1-5-a-1.png', filtered)
    cv2.imwrite('./output/ps1-5-a-2.png', edge_img)
    cv2.imwrite('./output/ps1-5-a-3.png', out_20)
    cv2.imwrite('./output/ps1-5-b-3.png', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def problem6():
    """Solution to part 5 of ps1."""
    # Load image.
    src = cv2.imread('./input/ps1-input2.png')
    # Create a grayscale image from original.
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # smooth image to find edges.
    smoothed = cv2.GaussianBlur(src_gray, (9, 9), 1)
    # extract edges using canny edge algorithm.
    edge_img = auto_canny(smoothed, 2)
    H = hough.hough_lines_acc(edge_img,)
    # Find peak points in Hough accumulator.
    peaks = hough.hough_peaks(H, edge_img.shape, 140)
    # perform histogram equalizaton to Hough accumulator to diplay it.
    H_ench = hough.enchance_acc(H)
    # Convert enchanced Hough image from gray to color to draw the peaks.
    H_ench = cv2.cvtColor(H_ench, cv2.COLOR_GRAY2RGB)
    src_gray = cv2.cvtColor(src_gray, cv2.COLOR_GRAY2RGB)
    # For each peak, draw a red dot in Hough accumulator array.
    H_peak = H_ench.copy()
    # Delete lines with 0 d.
    peaks = [(param, pixel) for param, pixel in peaks if param[0] > 1]
    for param, pixel in peaks:
        cv2.circle(H_peak, (pixel[1], pixel[0]), 1, (0, 0, 255), -1)
    # Draw found lines in the original image.
    line_img = hough.hough_lines_draw(src_gray, peaks)
    # Save the result.
    cv2.imwrite('./test/ps1-4-a-1.png', smoothed)
    cv2.imwrite('./test/ps1-4-b-1.png', edge_img)
    cv2.imwrite('./test/ps1-4-c-1.png', H_peak)
    cv2.imwrite('./test/ps1-4-c-2.png', line_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    # problem1()
    # problem2()
    # problem3()
    # problem4()
    # problem5()
    problem6()
