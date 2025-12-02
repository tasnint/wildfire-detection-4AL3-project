import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
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

# EfficientNet expects minimum 32x32 input, so we need to resize from 28x28
# Also convert grayscale to RGB (3 channels)
def resize_images(images, target_size=(32, 32)):
    """Resize images and convert to 3 channels"""
    resized = []
    for img in images:
        # Resize to target size
        img_resized = tf.image.resize(img[..., np.newaxis], target_size)
        # Convert to 3 channels by repeating
        img_rgb = tf.repeat(img_resized, 3, axis=-1)
        resized.append(img_rgb.numpy())
    return np.array(resized)

print("Resizing images to 32x32 for EfficientNet...")
x_train_full = resize_images(x_train_full)
x_test = resize_images(x_test)

# ===============================
# 2. Train/Validation Split
# ===============================
TRAINING_RATIO = 0.8

training_size = int(TRAINING_RATIO * len(x_train_full))
validation_size = len(x_train_full) - training_size

x_train, x_val = x_train_full[:training_size], x_train_full[training_size:]
y_train, y_val = y_train_full[:training_size], y_train_full[training_size:]

print(f"\nDataset sizes:")
print(f"Training size: {training_size}")
print(f"Validation size: {validation_size}")
print(f"Test size: {len(x_test)}")
print(f"Image shape: {x_train.shape[1:]}")

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ===============================
# 3. Build EfficientNet-B0 Model
# ===============================

def build_efficientnet_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Build EfficientNet-B0 model for Fashion-MNIST classification.
    
    EfficientNet uses:
    - Compound scaling (depth, width, resolution)
    - Mobile Inverted Bottleneck (MBConv) blocks
    - Squeeze-and-Excitation optimization
    - Swish activation function
    """
    # Load pre-trained EfficientNet-B0 (without top classification layer)
    # We use include_top=False to add our custom classifier
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',  # Use ImageNet pre-trained weights for transfer learning
        input_shape=input_shape,
        pooling='avg'  # Global Average Pooling
    )
    
    # Fine-tune strategy: freeze early layers, train later layers
    # EfficientNet-B0 has 237 layers, we'll freeze the first 80%
    freeze_until = int(len(base_model.layers) * 0.8)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    # Build complete model
    inputs = layers.Input(shape=input_shape)
    
    # EfficientNet expects normalized inputs in range [0, 1] (already done)
    x = base_model(inputs, training=False)
    
    # Add custom classification head
    x = layers.Dropout(0.5, name='dropout')(x)
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='EfficientNetB0_FashionMNIST')
    return model

# Build the model
model = build_efficientnet_model(input_shape=(32, 32, 3), num_classes=10)
model.summary()

print(f"\nTotal parameters: {model.count_params():,}")
print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# ===============================
# 4. Compile Model
# ===============================

# Adam optimizer with lower learning rate for fine-tuning
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Callbacks
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ===============================
# 5. Data Augmentation
# ===============================
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False
)

batch_size = 64

train_gen = train_datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)
val_gen = train_datagen.flow(x_val, y_val, batch_size=batch_size, shuffle=False)

# ===============================
# 6. Train Model
# ===============================
print("\nTraining EfficientNet-B0 on Fashion-MNIST...")
history = model.fit(
    train_gen,
    epochs=30,
    steps_per_epoch=len(x_train) // batch_size,
    validation_data=val_gen,
    validation_steps=len(x_val) // batch_size,
    callbacks=[lr_scheduler, early_stop]
)

# ===============================
# 7. Evaluate on Test Set
# ===============================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# ===============================
# 8. Detailed Metrics
# ===============================
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd

y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title("EfficientNet-B0 Confusion Matrix - Fashion-MNIST")
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
df_report.to_csv("results/efficientnet_fashion_mnist_report.csv", index=True)

# ===============================
# 9. Save Model and Visualize
# ===============================
os.makedirs("models", exist_ok=True)
model.save("models/efficientnet_b0_fashion_mnist.h5")

# Training curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('EfficientNet-B0 Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('EfficientNet-B0 Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("models/efficientnet_training_curves.png")
plt.show()

# ===============================
# 10. Sample Predictions
# ===============================
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
    
    # Show only one channel (they're all the same)
    axes[i].imshow(img[:, :, 0], cmap='gray')
    color = 'green' if pred_label == true_label else 'red'
    axes[i].set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}\nConf: {confidence:.2f}", 
                      color=color, fontsize=9)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("models/efficientnet_sample_predictions.png")
plt.show()

# ===============================
# 11. Compare Model Efficiency
# ===============================
print("\n==============================")
print(" EfficientNet-B0 Training Complete!")
print("==============================")
print(f" Test Accuracy: {test_acc:.4f}")
print(f" Total Parameters: {model.count_params():,}")
print(f" Trainable Parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
print("\n Model Efficiency Highlights:")
print(" • Uses compound scaling (depth + width + resolution)")
print(" • MBConv blocks with Squeeze-and-Excitation")
print(" • Transfer learning from ImageNet")
print(" • ~5.3M parameters (vs ResNet50's 25M)")
print("==============================")