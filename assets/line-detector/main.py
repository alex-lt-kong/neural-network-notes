import cv2
import numpy as np

image = cv2.imread('river.jpg')

kernel_vertical = np.array([[0, 1, 0],
                            [0, 1, 0],
                            [0, 1, 0],
                          ])
kernel_horizontal = np.array([
                              [0, 0, 0],
                              [1, 1, 1],
                              [0, 0, 0],
                            ])

cv2.imwrite('river-vertical.jpg', cv2.filter2D(src=image, ddepth=-1, kernel=kernel_vertical))
cv2.imwrite('river-horizontal.jpg', cv2.filter2D(src=image, ddepth=-1, kernel=kernel_horizontal))