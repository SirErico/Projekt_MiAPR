import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Load the gridmap data
os.chdir(os.path.dirname("/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/ros2_ws/src/mapr_rrt/maps/"))
map_file = "map.pgm"



with open(map_file, 'rb') as pgmf:
     grid_map = plt.imread(pgmf)
print("Grid map shape:", grid_map.shape) # (30, 30) - map
rows, cols = grid_map.shape

# with Image.open(map_file_test) as img:
#     grid_map_test = np.array(img)

# Normalize the input data
map_input = []
map_output = []
for i in range(grid_map.shape[0]):
    for j in range(grid_map.shape[1]):
        map_input.append([j / grid_map.shape[1], i / grid_map.shape[0]])
        map_output.append(1 if grid_map[i, j] == 0 else 0)

map_input = np.array(map_input)
map_output = np.array(map_output)

# Split the data into training and testing sets
# train_input, test_input, train_output, test_output = train_test_split(map_input, map_output, test_size=0.1, random_state=42)
# Not splitting the data for now
train_input = map_input
train_output = map_output



# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy',
    patience=200,
    min_delta=0.0,
    restore_best_weights=True
)


# Train the model
model.fit(train_input, train_output, epochs=5000, batch_size=8, verbose=2, callbacks=[early_stopping])
# verbose=2 for more detailed output, 1 for less detailed output, 0 for no output

# Evaluate the model
loss, accuracy = model.evaluate(train_input, train_output)
print("Loss:", loss)
print("Accuracy:", accuracy)




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

# Visualize the predictions
plt.figure(figsize=(10, 10))
plt.title("Predicted Occupancy Map")
plt.imshow(predictions, cmap='viridis', origin='upper')  # Use a colormap like 'viridis' for visualization
plt.colorbar(label="Occupancy Probability")
plt.show()


# Calculate the confusion matrix
preds = model.predict(train_input)
predicted = tf.squeeze(preds)
predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])
actual = np.array(train_output)
conf_mat = confusion_matrix(actual, predicted)
displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
displ.plot()
plt.show()


# Save the trained model
save_path = "/home/eryk/RiSA/sem1/MiAPR/Projekt_MiAPR/models/occupancy_model.keras"
if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save(save_path)
print("Model trained and saved!")