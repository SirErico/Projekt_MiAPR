import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import os

def main():
    # Load the gridmap data
    os.chdir(os.path.dirname("/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/ros2_ws/src/mapr_rrt/maps/"))
    map_file = "map_test_blurred.pgm"

    with open(map_file, 'rb') as pgmf:
        grid_map = plt.imread(pgmf)
    print("Grid map shape:", grid_map.shape) # (30, 30) - map
    rows, cols = grid_map.shape
    
    plt.figure(figsize=(20, 6))
    plt.subplot(1,2,1)
    plt.title("Original Grid Map (30x30)")
    plt.imshow(grid_map, cmap='viridis', origin='upper')
    plt.colorbar(label="Occupancy Value")
    
    
    
    csv_path = "/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/map_data_test_blurred.csv"
    # Load from CSV
    df = pd.read_csv(csv_path)
    map_input = df[['x', 'y']].values
    map_output = df['output'].values
    # Create a grid for visualization
    grid_size = 30  # Assuming a 30x30 grid
    grid = np.zeros((grid_size, grid_size))
    
    # Populate the grid with output values
    for i in range(len(map_input)):
        x = int(map_input[i, 0] * (grid_size - 1))
        y = int(map_input[i, 1] * (grid_size - 1))
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