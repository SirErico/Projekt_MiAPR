import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR = os.path.join(BASE_DIR, "ros2_ws/src/mapr_rrt/maps")
DATA_DIR = BASE_DIR

def main():
    # Load the gridmap data
    map_file = os.path.join(MAPS_DIR, "map_double_blurred.pgm")

    with open(map_file, 'rb') as pgmf:
        grid_map = plt.imread(pgmf)
    print("Grid map shape:", grid_map.shape) # (30, 30) - map
    rows, cols = grid_map.shape
    
    plt.figure(figsize=(20, 6))
    plt.subplot(1,2,1)
    plt.title("Original Grid Map (30x30)")
    plt.imshow(grid_map, cmap='viridis', origin='upper')
    plt.colorbar(label="Occupancy Value")
    
    
    
    csv_path = os.path.join(DATA_DIR, "map_data_double_blurred.csv")
    # Load from CSV
    df = pd.read_csv(csv_path)
    map_input = df[['x', 'y']].values
    map_output = df['output'].values
    # Create a grid for visualization
    grid_size = 30  # Assuming a 30x30 grid
    grid = np.zeros((grid_size, grid_size))
    
    # Populate the grid with output values
    for i in range(len(map_input)):
        # Match the same coordinate transformation as in sample_map.py
        x = np.clip(int(np.round(map_input[i, 0] * (grid_size - 1))), 0, grid_size - 1)
        y = np.clip(int(np.round(map_input[i, 1] * (grid_size - 1))), 0, grid_size - 1)
        grid[y, x] = map_output[i]
    
    # Display the map
    plt.subplot(1,2,2)
    plt.imshow(grid, cmap='viridis', origin='upper')
    plt.colorbar(label='Output Value')
    plt.title('Map from CSV Data')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()
    
    
    
if __name__ == "__main__":
    main()