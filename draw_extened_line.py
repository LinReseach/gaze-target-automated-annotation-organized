
    
import cv2

def extend_line_to_boundary(image, x1, y1, x2, y2):
    # Get image dimensions
    height, width, _ = image.shape

    # Calculate slope of the line
    if x2 - x1 != 0:
        slope = (y2 - y1) / (x2 - x1)
    else:
        slope = float('inf')

    # Calculate y-intercept
    y_intercept = y1 - slope * x1

    # Calculate new points to extend the line to image boundaries
    if slope != 0:
        # Extend to top boundary (y = 0)
        x_top = int(-y_intercept / slope)
        y_top = 0

        # Extend to bottom boundary (y = height - 1)
        x_bottom = int((height - 1 - y_intercept) / slope)
        y_bottom = height - 1
    else:
        # For vertical lines, x coordinates remain the same
        x_top = x1
        x_bottom = x1

        # Extend to top and bottom boundaries
        y_top = 0
        y_bottom = height - 1

    # Draw the extended line on the image (in this example, in red)
   # cv2.line(image, (x1, y1), (x_top, y_top), (0, 0, 255), 2)
    cv2.line(image, (x1, y1), (x_bottom, y_bottom), color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    return image


if __name__ == "__main__":
    # Load your image
    image = cv2.imread('/Users/chenglinlin/Documents/annotation/other dataset/Toddler-study/frames/40_2_front/02836.jpg')

    # Points (x1, y1) and (x2, y2)
    x1, y1 = 100, 100
    x2, y2 = 300, 200

    # Call the function to extend the line
    image_with_line = extend_line_to_boundary(image, x1, y1, x2, y2)

    # Display the image with the extended line
    cv2.imshow("Extended Line", image_with_line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
