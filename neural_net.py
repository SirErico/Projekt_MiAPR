import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib
import pandas as pd
import os


def neural_net():
    # Load the gridmap data
    os.chdir(os.path.dirname("/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/ros2_ws/src/mapr_rrt/maps/"))
    map_file = "map.pgm"

    with open(map_file, 'rb') as pgmf:
        grid_map = plt.imread(pgmf)
    rows, cols = grid_map.shape
    
    csv_path = "/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/map_data.csv"
    # Load from CSV
    df = pd.read_csv(csv_path)
    train_input = df[['x', 'y']].values
    train_output = df['output'].values
    print(train_input.shape)
    print(train_output.shape)

    # Split the data into training and testing sets
    # train_input, test_input, train_output, test_output = train_test_split(map_input, map_output, test_size=0.1, random_state=42)
    # Not splitting the data for now
    # train_input = map_input
    # train_output = map_output



    # # Define the neural network architecture
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Dense(128, activation='relu', input_shape=(2,)),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    # ])
    
    # Define the neural network architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='tanh', input_shape=(2,)),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])



    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

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


    # Init early stop
    callback_acc = StopAtAccuracy(target_acc = 0.98)
    # callback_loss = StopAtLoss(target_loss=0.005)
    # Train the model
    model.fit(train_input, train_output, epochs=6000, batch_size=256, verbose=2, callbacks=[callback_acc])
    # verbose=2 for more detailed output, 1 for less detailed output, 0 for no output

    # Evaluate the model
    loss, accuracy = model.evaluate(train_input, train_output)
    print("Loss:", loss)
    print("Accuracy:", accuracy)

    # Save the trained model
    save_path = "/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/models/occupancy_model_10.keras"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print("Model trained and saved!")


    test_map_input = train_input

    print("Now predicting on train data!")
    # Predict occupancy for the entire map
    predictions = model.predict(test_map_input)
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


    # # Calculate the confusion matrix
    # preds = model.predict(train_input)
    # predicted = tf.squeeze(preds)
    # predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])
    # actual = np.array(train_output)
    # conf_mat = confusion_matrix(actual, predicted)
    # displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    # displ.plot()
    # plt.show()
    
if __name__ == '__main__':
    neural_net()
    
    
    
'''
occupancy_model_5 -> new 1000x1000 neural net, acc 98% b_size = 128
occupancy_model_6 -> new nn, 90% acc, b_size = 64 
SHOULD USE BIGGER BATCH SIZE
occupancy_model_7 -> 0.9 acc, b_size = 256 - also pretty bad
occupancy_model_8 -> 0.96 acc, b_size = 256
occupancy_model_9 -> tanh activation func, 0.98 acc, b_size = 256
'''