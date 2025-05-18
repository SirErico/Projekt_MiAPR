import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING
import tensorflow as tf
from tensorflow import keras
import sys
import matplotlib.pyplot as plt


def main():
    # Load the gridmap data
    os.chdir(os.path.dirname("/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/ros2_ws/src/mapr_rrt/maps/"))
    map_file = "map.pgm"

    with open(map_file, 'rb') as pgmf:
        grid_map = plt.imread(pgmf)
    print("Grid map shape:", grid_map.shape) # (30, 30) - map
        
    # Load the model
    model_path = '/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/models/occupancy_model_3.keras'
    model = tf.keras.models.load_model(model_path)
    
    
    # Test the model on the entire map
    height, width = grid_map.shape
    test_map_input = []

    # Generate normalized input for every point in the map
    for i in range(height):
        for j in range(width):
            test_map_input.append([j / width, i / height])

    test_map_input = np.array(test_map_input)
    print(test_map_input.shape)


if __name__ == "__main__":
    main()