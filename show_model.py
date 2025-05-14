import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
def show_model():
    # Load the gridmap data
    os.chdir(os.path.dirname("/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/ros2_ws/src/mapr_rrt/maps/"))
    map_file = "map.pgm"



    with open(map_file, 'rb') as pgmf:
        grid_map = plt.imread(pgmf)
    print("Grid map shape:", grid_map.shape) # (30, 30) - map
    rows, cols = grid_map.shape
        

    model_path = '/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/models/occupancy_model_1.keras'
    model = tf.keras.models.load_model(model_path)

    # Test the model on the entire map
    height, width = grid_map.shape
    test_map_input = []

    # Generate normalized input for every point in the map
    for i in range(height):
        for j in range(width):
            test_map_input.append([j / width, i / height])

    test_map_input = np.array(test_map_input)


    # Predict occupancy for the entire map
    predictions = model.predict(test_map_input)
    predictions = predictions.reshape((height, width))  # Reshape predictions to match the map dimensions

    plt.figure(figsize=(20, 10))
    plt.subplot(1,2,1)
    plt.title("Original Grid Map")
    plt.imshow(grid_map, cmap='viridis', origin='upper')
    plt.colorbar(label="Occupancy Value")
    # Visualize the predictions
    plt.subplot(1,2,2)
    plt.title("Predicted Occupancy Map")
    plt.imshow(predictions, cmap='viridis', origin='upper')  # Use a colormap like 'viridis' for visualization
    plt.colorbar(label="Occupancy Probability")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    show_model()