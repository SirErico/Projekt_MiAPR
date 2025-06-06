import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR = os.path.join(BASE_DIR, "ros2_ws/src/mapr_rrt/maps")
DATA_DIR = BASE_DIR
MODELS_DIR = os.path.join(BASE_DIR, "models")

def neural_net():
    # Load the gridmap data
    map_file = os.path.join(MAPS_DIR, "map_double_blurred.pgm")
    with open(map_file, 'rb') as pgmf:
        grid_map = plt.imread(pgmf)
    rows, cols = grid_map.shape
    
    csv_path = os.path.join(DATA_DIR, "map_data_double_blurred.csv")
    # Load from CSV
    df = pd.read_csv(csv_path)
    map_input = df[['x', 'y']].values
    map_output = df['output'].values
    train_input = map_input
    train_output = map_output

    # Split the data into training and testing sets
    # train_input, test_input, train_output, test_output = train_test_split(map_input, map_output, test_size=0.1, random_state=42, stratify=map_output)

    
    # NEURAL NET ARCHITECTURE FOR TEST_BLURRED I MAP
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Dense(16, activation='tanh', input_shape=(2,)),
    #     tf.keras.layers.Dense(16, activation='tanh'),
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    # ])


    # Compile the model
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    # no metrics
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Early Stoppage of training at given accuracy
    class StopAtAccuracy(tf.keras.callbacks.Callback):
        def __init__(self, target_acc):
            super().__init__()
            self.target_acc = target_acc

        def on_epoch_end(self, epoch, logs):
            acc = logs.get("accuracy")
            if acc is not None and acc >= self.target_acc:
                print(f"\nStopping training: reached {acc:.2f} accuracy at epoch {epoch + 1}")
                self.model.stop_training = True
                
    # Early Stoppage of training at given accuracy
    class StopAtLoss(tf.keras.callbacks.Callback):
        def __init__(self, target_loss):
            super().__init__()
            self.target_loss = target_loss

        def on_epoch_end(self, epoch, logs):
            loss_ = logs.get("loss")
            if loss_ is not None and loss_ <= self.target_loss:
                print(f"\nStopping training: reached {loss_:.2f} loss at epoch {epoch + 1}")
                self.model.stop_training = True
    
    model.fit(
        train_input,
        train_output,
        epochs=6000,
        batch_size=64,
        verbose=2,
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=40,
                restore_best_weights=True,
                min_delta=1e-5
            )
        ]
    )
    # verbose=2 for more detailed output, 1 for less detailed output, 0 for no output

    # Evaluate on both train and test sets
    results = model.evaluate(train_input, train_output)
    # test_loss, test_accuracy = model.evaluate(test_input, test_output)


    print("\nTraining Results:")
    # print("Train Loss:", train_loss)
    # print("Train Accuracy:", train_accuracy)
    # print("\nTesting Results:")
    # print("Test Loss:", test_loss)
    # print("Test Accuracy:", test_accuracy)
    print("RESULTS: ", results)

    # Save the trained model
    save_path = os.path.join(MODELS_DIR, "occupancy_model_double1.keras")
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(save_path)
    print(f"Model trained and saved to: {save_path}")


    test_map_input = map_input
    print("Now predicting on train data!")
    # Predict occupancy for the entire map
    predictions = model.predict(test_map_input)
    print("Min/Max prediction:", np.min(predictions), np.max(predictions))
    predictions = predictions.reshape((1000, 1000))  # Reshape predictions to match the map dimensions

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
    
    preds = model.predict(train_input).flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(train_output, preds, alpha=0.1, s=1)
    plt.xlabel("True Occupancy")
    plt.ylabel("Predicted Occupancy")
    plt.title("True vs Predicted")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

if __name__ == '__main__':
    neural_net()
    
    
    
'''
occupancy_model_5 -> new 1000x1000 neural net, acc 98% b_size = 128
occupancy_model_6 -> new nn, 90% acc, b_size = 64 
SHOULD USE BIGGER BATCH SIZE
occupancy_model_7 -> 0.9 acc, b_size = 256 - also pretty bad
occupancy_model_8 -> 0.96 acc, b_size = 256
occupancy_model_9 -> tanh activation func, 0.98 acc, b_size = 256

mniejsza siec + mapa prawdopodobienstw
train test split + parametr stratify
przesuniecie punktu
rrt vertices
'''