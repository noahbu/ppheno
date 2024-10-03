import cv2
import numpy as np

# Load the cropped coin image
image = cv2.imread('/Users/noahbucher/Documents_local/Plant_reconstruction/ppheno/data/leaf_area/gt/blue_leaf/Scan_1_c.png')

# List to store the points clicked by the user
points = []

# Mouse callback function to capture two points
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add the clicked point to the list
        points.append((x, y))

        # Draw a small circle on the image where clicked
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", image)

        # If two points are clicked, calculate the distance
        if len(points) == 2:
            # Compute the Euclidean distance between the two points
            point1, point2 = points
            distance_pixels = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            print(f"Distance in pixels: {distance_pixels:.2f} pixels")
            
            # Convert pixels to mm using the DPI information (300 DPI -> 11.81 pixels per mm)
            pixels_per_mm = 11.81  # Calculated from 300 DPI
            distance_mm = distance_pixels / pixels_per_mm
            print(f"Measured diameter in mm: {distance_mm:.2f} mm")

            # Show the result on the image
            cv2.line(image, point1, point2, (255, 0, 0), 2)
            cv2.imshow("Image", image)

# Display the image and set the callback function for mouse events
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", click_event)

# Wait until a key is pressed, then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
