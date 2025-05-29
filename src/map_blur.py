import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Set the working directory and file path
    base_dir = "/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/ros2_ws/src/mapr_rrt/maps/"
    os.chdir(os.path.dirname(base_dir))
    map_file = "map_test.pgm"
    
    # Read the image
    img = cv.imread(map_file, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image from {map_file}")
        return
    
    # Apply Gaussian blur
    blurred = cv.GaussianBlur(img, (5, 5), 0)
    
    output_file = os.path.join(base_dir, "map_test_blurred.pgm")
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

