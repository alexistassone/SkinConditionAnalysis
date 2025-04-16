import tensorflow as tf
import os
import Data_Preprocessing as dp

# Load data from preprocessing
train_data, test_data = dp.load_and_preprocess_data()

# Build the model using CNN
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(256, 256, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(train_data.class_names), activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Check the number of steps per epoch based on the number of batches
steps_per_epoch = len(train_data)

# Train the model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data,
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data)

# Save the model
model_dir = os.path.join(os.path.dirname(__file__), '..', 'Model')
model.save(model_dir)

def main():
    print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')
    print("Model saved successfully.")