import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR = os.path.join(BASE_DIR, "ros2_ws/src/mapr_rrt/maps")
DATA_DIR = BASE_DIR

def main():
    map_file = os.path.join(MAPS_DIR, "map_double.pgm")

    
    # Read the image
    img = cv.imread(map_file, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image from {map_file}")
        return
    
    # Apply Gaussian blur
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    
    output_file = os.path.join(MAPS_DIR, "map_double_blurred.pgm")
    cv.imwrite(output_file, blurred)
    # Display original and blurred images
    plt.figure(figsize=(20, 6))
    plt.title("Original Grid Map (30x30)")
    plt.imshow(blurred, cmap='viridis', origin='upper')
    plt.colorbar(label="Occupancy Value")
    plt.show()
    
    # Wait for key press and close windows
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

