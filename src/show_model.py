import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import pandas as pd
import os

def show_model():
    csv_path = "/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/map_data_test_blurred.csv"
    df = pd.read_csv(csv_path)
    test_map_input = df[['x', 'y']].values
    
    # Load the grid map
    os.chdir(os.path.dirname("/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/ros2_ws/src/mapr_rrt/maps/"))
    map_file = "map_test_blurred.pgm"
    with open(map_file, 'rb') as pgmf:
        grid_map = plt.imread(pgmf)
    print("Grid map shape:", grid_map.shape)  # Should be (30, 30)
    rows, cols = grid_map.shape

    model_path = '/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/models/occupancy_model_test_blurred2.keras'
    model = tf.keras.models.load_model(model_path)

    predictions = model.predict(test_map_input)
    predictions = predictions.reshape((1000, 1000))

    # === Compute gradients on 30x30 grid ===
    width, height = 30, 30

    DX = np.zeros((height, width))
    DY = np.zeros((height, width))

    def compute_gradient(x, y):
        input_tensor = tf.convert_to_tensor([[x / cols, y / rows]], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            pred = model(input_tensor)
        grad = tape.gradient(pred, input_tensor).numpy()[0]
        dx = grad[0] * cols
        dy = grad[1] * rows
        return dx, dy

    for i in range(height):
        for j in range(width):
            dx, dy = compute_gradient(j, i)  # (x, y) = (col, row)
            DX[i, j] = dx
            DY[i, j] = dy

    # === Plot everything ===
    plt.figure(figsize=(18, 6))

    # Original map
    plt.subplot(1, 2, 1)
    plt.title("Original Grid Map (30x30)")
    plt.imshow(grid_map, cmap='viridis', origin='upper')
    plt.colorbar(label="Occupancy Value")

    # Predictions
    plt.subplot(1, 2, 2)
    plt.title("NN Predictions (1000x1000)")
    plt.imshow(predictions, cmap='viridis', origin='upper')
    plt.colorbar(label="Occupancy Probability")
    plt.show()
    
    # === Gradient Text Overlay Plot ===
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Gradient Vectors (text overlay)")
    im = ax.imshow(grid_map, cmap='gray', origin='upper', extent=[0, width, height, 0])
    ax.set_aspect('equal')  # Maintain square cells

    for i in range(height):
        for j in range(width):
            dx = DX[i, j]
            dy = DY[i, j]
            text = f"{dx:+.1f}\n{dy:+.1f}"
            ax.text(j + 0.5, i + 0.5, text, fontsize=6, color='red',
                    ha='center', va='center', linespacing=1.0)

    fig.colorbar(im, ax=ax, label="Background Occupancy")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    show_model()
