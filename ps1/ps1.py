"""Implementation of problem set 1."""
import cv2
import hough

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



if __name__ == '__main__':
    # problem1()
    problem2()
