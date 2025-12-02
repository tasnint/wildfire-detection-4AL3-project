import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# ===============================
# 1. Load Fashion-MNIST Dataset
# ===============================
print("Loading Fashion-MNIST dataset...")
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to [0, 1]
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add channel dimension (Fashion-MNIST is grayscale)
x_train_full = np.expand_dims(x_train_full, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# ===============================
# 2. Train/Validation Split
# ===============================
TRAINING_RATIO = 0.8

training_size = int(TRAINING_RATIO * len(x_train_full))
validation_size = len(x_train_full) - training_size

x_train, x_val = x_train_full[:training_size], x_train_full[training_size:]
y_train, y_val = y_train_full[:training_size], y_train_full[training_size:]

print(f"\nDataset sizes:")
print(f"Full training dataset: {len(x_train_full)}")
print(f"Training size after split (80%): {training_size}")
print(f"Validation size after split (20%): {validation_size}")
print(f"Test dataset size: {len(x_test)}")

# Class names for Fashion-MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ===============================
# 3. Data Augmentation (Optional)
# ===============================
# Create data generators with augmentation for training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False  # Fashion items shouldn't be flipped horizontally
)

# No augmentation for validation/test
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

batch_size = 64

# Create generators
train_gen = train_datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)
val_gen = val_datagen.flow(x_val, y_val, batch_size=batch_size, shuffle=False)
test_gen = test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)

# ===============================
# 4. CNN Model Architecture
# ===============================
# Input shape: 28x28x1 (Fashion-MNIST images)
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Dropout(0.3),

    # ============================================
    # Global Average Pooling layer
    # Reduces parameters and improves generalization
    # ============================================
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    
    # Final output layer for 10 classes (multi-class classification)
    layers.Dense(10, activation='softmax')
])

# ============================================
# OPTIMIZER + CALLBACKS (Adaptive Learning Rate)
# ============================================

# Adam optimizer with adaptive learning rate
optimizer = tf.keras.optimizers.Adam()

# Reduce learning rate when validation loss plateaus
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# Early stopping to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Compile model with categorical crossentropy (multi-class classification)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',  # sparse because labels are integers
    metrics=['accuracy']
)

model.summary()

# ===============================
# 5. Train Model
# ===============================
print("\nTraining model...")
history = model.fit(
    train_gen,
    epochs=20,
    steps_per_epoch=len(x_train) // batch_size,
    validation_data=val_gen,
    validation_steps=len(x_val) // batch_size,
    callbacks=[lr_scheduler, early_stop]
)

# ===============================
# 6. Evaluate on Test Set
# ===============================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# ===============================
# 6.1 Detailed Performance Metrics
# ===============================
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd

# Get predictions
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title("Confusion Matrix - Fashion-MNIST Test Set")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Classification report
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()

print("\n===== Classification Report =====")
print(df_report.round(3))

# Save results
os.makedirs("results", exist_ok=True)
df_report.to_csv("results/fashion_mnist_classification_report.csv", index=True)

# ===============================
# 7. Save Model and Training Curves
# ===============================
os.makedirs("models", exist_ok=True)
model.save("models/fashion_mnist_cnn_model.h5")

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("models/fashion_mnist_training_curves.png")
plt.show()

# ===============================
# 8. Visualize Sample Predictions
# ===============================
# Show some test predictions
num_samples = 10
indices = np.random.choice(len(x_test), num_samples, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i, idx in enumerate(indices):
    img = x_test[idx]
    true_label = y_test[idx]
    pred_probs = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
    pred_label = np.argmax(pred_probs)
    confidence = pred_probs[pred_label]
    
    axes[i].imshow(img.squeeze(), cmap='gray')
    color = 'green' if pred_label == true_label else 'red'
    axes[i].set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}\nConf: {confidence:.2f}", 
                      color=color, fontsize=9)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("models/sample_predictions.png")
plt.show()

print("\n==============================")
print(" Training Complete!")
print(f" Test Accuracy: {test_acc:.4f}")
print(f" Model saved to: models/fashion_mnist_cnn_model.h5")
print("==============================")