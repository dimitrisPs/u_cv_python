"""Implementation of problem set 1."""
import cv2


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


if __name__ == '__main__':
    problem1()
