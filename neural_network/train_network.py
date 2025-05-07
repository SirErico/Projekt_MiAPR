# Importing necessary modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, LeakyReLU
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


os.chdir("ros2_ws/src/mapr_rrt/maps")

X = [[]]

#Y = 

f = "map_medium.pgm"

with open(f, 'rb') as pgmf:
     im = plt.imread(pgmf)

# Get array shape
rows, cols = im.shape

# Generate all possible (x, y) coordinates
x_coords, y_coords = np.indices((rows, cols))

# Flatten the coordinate arrays
x = x_coords.flatten()
y = y_coords.flatten()
# Get values at each coordinate
values = im[x, y]

# Combine x and y into a coordinate vector (optional)
coordinates = np.column_stack((x, y))


values = np.array([im[x, y] for x, y in coordinates])

values = values/255
coordinates = coordinates/29

print("Coordinates:\n", coordinates)
print("Values:\n", values[31])

X_train, X_test, y_train, y_test = train_test_split(coordinates, values, test_size=0.33, random_state=42)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)



model = Sequential([
    Input(shape=(2,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])



model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1000, 
          batch_size=2000, 
          validation_split=0.2,
          class_weight=dict(enumerate(class_weights)))


results = model.evaluate(X_test, y_test, verbose=0)
print('Test loss, Test accuracy:', results)

preds = model.predict(X_test)

predicted = tf.squeeze(preds)
predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])
actual = np.array(y_test)
conf_mat = confusion_matrix(actual, predicted)
displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
displ.plot()
plt.show()


# Load MNIST dataset
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()