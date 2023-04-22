import cv2
import numpy as np


image = cv2.imread('valve-original.png')

sobel_kernel = np.array([
                          [1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1],
                        ])


cv2.imwrite('valve-convolved.png', cv2.filter2D(src=image, ddepth=-1, kernel=sobel_kernel))
