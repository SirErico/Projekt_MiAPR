import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import os

def show_model():
    csv_path = "/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/map_data.csv"
    # Load from CSV
    df = pd.read_csv(csv_path)
    test_map_input = df[['x', 'y']].values
    train_output = df['output'].values
    
    # Load the gridmap data
    os.chdir(os.path.dirname("/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/ros2_ws/src/mapr_rrt/maps/"))
    map_file = "map.pgm"

    with open(map_file, 'rb') as pgmf:
        grid_map = plt.imread(pgmf)
    print("Grid map shape:", grid_map.shape) # (30, 30) - map
    rows, cols = grid_map.shape
        
    model_path = '/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/models/occupancy_model_test1.keras'
    model = tf.keras.models.load_model(model_path)

    # Predict occupancy for the entire map
    predictions = model.predict(test_map_input)
    predictions = predictions.reshape((1000, 1000))  # Reshape predictions to match the sampling dimensions

    # Create a figure with three subplots
    plt.figure(figsize=(20, 6))
    
    # Original grid map (30x30)
    plt.subplot(1, 2, 1)
    plt.title("Original Grid Map (30x30)")
    plt.imshow(grid_map, cmap='viridis', origin='upper')
    plt.colorbar(label="Occupancy Value")
    
    # Neural network predictions (1000x1000)
    plt.subplot(1, 2, 2)
    plt.title("Neural Network Predictions (1000x1000)")
    plt.imshow(predictions, cmap='viridis', origin='upper')
    plt.colorbar(label="Occupancy Probability")
    
    # Difference map
    # plt.subplot(1, 3, 3)
    # plt.title("Difference Map")
    # # Resize original map to match predictions using proper scaling
    # scale_factor = 1000 // 30
    # resized_original = np.repeat(np.repeat(grid_map, scale_factor, axis=0), scale_factor, axis=1)
    # # Ensure the resized map is exactly 1000x1000
    # if resized_original.shape != (1000, 1000):
    #     resized_original = resized_original[:1000, :1000]
    # difference = np.abs(predictions - (1.0 - resized_original/255.0))
    # plt.imshow(difference, cmap='hot', origin='upper')
    # plt.colorbar(label="Absolute Difference")
    
    # plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    show_model()