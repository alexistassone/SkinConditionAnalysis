import tensorflow as tf
import os

# Define paths
train_dir = os.path.join(os.path.dirname(__file__), '..', 'Data', 'SkinDisease', 'train')
test_dir = os.path.join(os.path.dirname(__file__), '..', 'Data', 'SkinDisease', 'test')

# Load train data
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(256, 256),
    batch_size=32,
    label_mode='int'
)

# Load test data
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(256, 256),
    batch_size=32,
    label_mode='int'
)

# Print class names
class_names = train_data.class_names
print("Class Names :", class_names)

# Prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
test_data = test_data.prefetch(buffer_size=AUTOTUNE)

# Check batch shapes
for images, labels in train_data.take(1): # Check one batch
    print("Batch Shape:", images.shape)
    print("Label Shape:", labels.shape)