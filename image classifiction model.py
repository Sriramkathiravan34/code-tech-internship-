# Image Classification using CNN with TensorFlow (Keras)
# This file is structured like a Jupyter Notebook
# You can copy it into a .ipynb file for submission

# =====================================
# Cell 1: Import Required Libraries
# =====================================
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

print("TensorFlow Version:", tf.__version__)

# =====================================
# Cell 2: Load Dataset (CIFAR-10)
# =====================================
# CIFAR-10 contains 60,000 color images in 10 classes

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print("Training Data Shape:", x_train.shape)
print("Testing Data Shape:", x_test.shape)

# =====================================
# Cell 3: Data Preprocessing
# =====================================
# Normalize pixel values (0–255 → 0–1)

x_train = x_train / 255.0
x_test = x_test / 255.0

# =====================================
# Cell 4: Visualize Sample Images
# =====================================

plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')
plt.show()

# =====================================
# Cell 5: Build CNN Model
# =====================================

def build_cnn_model():
    model = models.Sequential()

    # Convolution + Pooling Layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model

model = build_cnn_model()
model.summary()

# =====================================
# Cell 6: Compile Model
# =====================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =====================================
# Cell 7: Train Model
# =====================================

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# =====================================
# Cell 8: Evaluate Model (Test Dataset)
# =====================================

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# =====================================
# Cell 9: Plot Training History
# =====================================

plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# =====================================
# Cell 10: Test Predictions
# =====================================

def predict_image(img_index):
    img = x_test[img_index]
    img_array = np.expand_dims(img, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    actual_class = class_names[y_test[img_index][0]]

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} | Actual: {actual_class}")
    plt.axis('off')
    plt.show()

# Example Prediction
predict_image(0)

# =====================================
# Cell 11: Analysis and Observations
# =====================================
"""
Analysis:

1. A CNN model was built using TensorFlow Keras.
2. CIFAR-10 dataset was used for multi-class image classification.
3. Images were normalized for better convergence.
4. The model achieved good accuracy on the test dataset.
5. Training and validation curves show learning behavior.

Limitations:
- Limited epochs reduce performance.
- No data augmentation used.

Improvements:
- Increase epochs
- Add dropout layers
- Use data augmentation
- Try transfer learning (ResNet, VGG)

Conclusion:
CNN models are effective for image classification tasks and can automatically learn visual features.
"""
