import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib
from PIL import Image
import pandas as pd
import os



def sample_data():
    # Load the gridmap data
    os.chdir(os.path.dirname("/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/ros2_ws/src/mapr_rrt/maps/"))
    map_file = "map_test_blurred.pgm"

    with open(map_file, 'rb') as pgmf:
        grid_map = plt.imread(pgmf)
    print("Grid map shape:", grid_map.shape)  # (30, 30) - map
    rows, cols = grid_map.shape
    print(rows, cols)

    # Create a grid of normalized coordinates
    x = np.linspace(0, 1, num=1000)
    y = np.linspace(0, 1, num=1000)
    X, Y = np.meshgrid(x, y)

    # Flatten and stack coordinates
    map_input = np.column_stack((X.ravel(), Y.ravel()))

    # Convert normalized coordinates to grid indices
    grid_x = np.clip(np.round(X * (cols - 1)), 0, cols - 1).astype(int)
    grid_y = np.clip(np.round(Y * (rows - 1)), 0, rows - 1).astype(int)

    # Get output values directly using the indices
    map_output = 1.0 - (grid_map[grid_y, grid_x] / 255.0).ravel()

    print("Map input shape:", map_input.shape)  # Should be (1000000, 2)
    print("Map output shape:", map_output.shape)  # Should be (1000000,)

    # Create a DataFrame
    df = pd.DataFrame(map_input, columns=['x', 'y'])
    df['output'] = map_output

    # Save to CSV
    csv_path = "/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/map_data_test_blurred.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    
    plt.hist(df['output'], bins=50)
    plt.title("Occupancy value distribution")
    plt.xlabel("Occupancy (0 = free, 1 = occupied)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()
   
if __name__ == '__main__':
    sample_data()