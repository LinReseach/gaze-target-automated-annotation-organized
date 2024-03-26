#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:41:31 2024

@author: chenglinlin
"""

import cv2
import numpy as np

def draw_arrow_to_boundary(image, x1, y1, x2, y2):
    # Calculate the slope of the line
    slope = (y2 - y1) / (x2 - x1)

    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate the extended endpoint to the boundary of the image
    if x2 > x1:
        x3 = width - 1
    else:
        x3 = 0
    y3 = int(y2 + slope * (x3 - x2))

    # Calculate the length of the arrow
    arrow_length = min(50, np.sqrt((x3 - x2)**2 + (y3 - y2)**2))

    # Calculate the angle of the arrow
    angle = np.arctan2(y2 - y1, x2 - x1)

    # Calculate the coordinates of the arrowhead
    x4 = int(x2 + arrow_length * np.cos(angle))
    y4 = int(y2 + arrow_length * np.sin(angle))

    # Draw the line from (x1, y1) to (x3, y3)
    cv2.line(image, (x1, y1), (x3, y3), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)



if __name__ == "__main__":
    # Load the image
    image = cv2.imread('/Users/chenglinlin/Documents/annotation/other dataset/Toddler-study/frames/40_2_front/02836.jpg')

    # Define the points (x1, y1), (x2, y2)
    x1, y1 = 50, 50
    x2, y2 = 30, 30

    # Call the function to draw the arrow to the boundary
    draw_arrow_to_boundary(image, x1, y1, x2, y2)

    # Display the image with the arrow
    cv2.imshow("Arrow to Boundary", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
