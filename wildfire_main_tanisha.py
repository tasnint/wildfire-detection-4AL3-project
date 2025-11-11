import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {gpus}")
    # Only enable mixed precision if GPU is available
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled (float16)")
else:
    print("No GPU detected - running on CPU with float32")

# ===============================
# 1. Directory Paths
# ===============================
base_dir = "data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# ===============================
# 2. Configuration
# ===============================
IMG_SIZE = (128, 128)  # Start with 128x128, can try 224x224 later
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

# ===============================
# 3. Data Augmentation Function
# ===============================
def augment_image(image, label):
    """Apply data augmentation to training images."""
    # Random rotation
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # Ensure values stay in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

# ===============================
# 4. Load Data with tf.data API
# ===============================
print("Loading training data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=True,
    seed=123
)

print("Loading validation data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False,
    seed=123
)

print("Loading test data...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False,
    seed=123
)

# ===============================
# 5. Optimize Data Pipeline
# ===============================
# Normalize images to [0, 1]
normalization_layer = layers.Rescaling(1./255)

# Apply normalization and augmentation
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(augment_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache()  # Cache after augmentation for speed
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

# Validation and test only need normalization
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.cache()
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache()
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

print("\nData pipeline optimized with caching and prefetching!")

# ===============================
# 6. CNN Model Architecture
# ===============================
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', 
                  input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Dropout(0.3),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid', dtype='float32')
])

# Use a simple fixed learning rate (callback will adjust it)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy', 
    metrics=['accuracy']
)
model.summary()

# ===============================
# 7. Callbacks for Efficient Training
# ===============================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'models/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ===============================
# 8. Train Model
# ===============================
print("\nStarting training...")
history = model.fit(
    train_ds, 
    epochs=20,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

# ===============================
# 9. Evaluate on Test Set
# ===============================
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# ===============================
# 10. Save Model and Training Curves
# ===============================
os.makedirs("models", exist_ok=True)
model.save("models/wildfire_cnn_final.keras")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train acc', marker='o')
plt.plot(history.history['val_accuracy'], label='val acc', marker='s')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss', marker='o')
plt.plot(history.history['val_loss'], label='val loss', marker='s')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("models/training_curves.png", dpi=150)
plt.show()

# ===============================
# 11. Predict on Sample Images
# ===============================
print("\nTesting predictions on sample images...")

# Get a batch from test set
for images, labels in test_ds.take(1):
    predictions = model.predict(images)
    
    # Show first 5 predictions
    for i in range(min(5, len(predictions))):
        pred_value = predictions[i][0]
        pred_label = "Fire" if pred_value > 0.5 else "No Fire"
        true_label = "Fire" if labels[i] == 1 else "No Fire"
        print(f"Image {i+1}: Predicted={pred_label} ({pred_value:.3f}), True={true_label}")

print("\n==============================")
print("Training Complete!")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Final test accuracy: {test_acc:.4f}")
print("Model saved to: models/wildfire_cnn_final.keras")
print("==============================\n")