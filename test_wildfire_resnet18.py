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

# Add channel dimension and convert grayscale to RGB (ResNet expects 3 channels)
# We replicate the grayscale channel 3 times
x_train_full = np.repeat(x_train_full[..., np.newaxis], 3, axis=-1)
x_test = np.repeat(x_test[..., np.newaxis], 3, axis=-1)

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

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ===============================
# 3. ResNet18 Building Blocks
# ===============================

def basic_block(x, filters, stride=1, name=None):
    """
    Basic ResNet block with two 3x3 convolutions.
    This is the building block used in ResNet18 and ResNet34.
    """
    # Save input for skip connection
    shortcut = x
    
    # First conv layer
    x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same', 
                      name=f'{name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Activation('relu', name=f'{name}_relu1')(x)
    
    # Second conv layer
    x = layers.Conv2D(filters, (3, 3), strides=1, padding='same', 
                      name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    
    # Adjust shortcut if dimensions changed
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, 
                                 padding='same', name=f'{name}_shortcut_conv')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)
    
    # Add skip connection
    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.Activation('relu', name=f'{name}_relu2')(x)
    
    return x

def resnet_stage(x, filters, num_blocks, stride, stage_name):
    """
    A ResNet stage consists of multiple basic blocks.
    """
    # First block may have stride > 1 for downsampling
    x = basic_block(x, filters, stride=stride, name=f'{stage_name}_block1')
    
    # Remaining blocks have stride=1
    for i in range(1, num_blocks):
        x = basic_block(x, filters, stride=1, name=f'{stage_name}_block{i+1}')
    
    return x

# ===============================
# 4. Build ResNet18 Model
# ===============================

def build_resnet18(input_shape=(28, 28, 3), num_classes=10):
    """
    Build ResNet18 architecture:
    - Initial conv layer
    - 4 stages with [2, 2, 2, 2] blocks
    - Global Average Pooling
    - Dense output layer
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution (modified for 28x28 input - no maxpool to preserve size)
    x = layers.Conv2D(64, (3, 3), strides=1, padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    
    # ResNet18 has 4 stages with [2, 2, 2, 2] blocks
    # Stage 1: 64 filters, 2 blocks
    x = resnet_stage(x, filters=64, num_blocks=2, stride=1, stage_name='stage1')
    
    # Stage 2: 128 filters, 2 blocks
    x = resnet_stage(x, filters=128, num_blocks=2, stride=2, stage_name='stage2')
    
    # Stage 3: 256 filters, 2 blocks
    x = resnet_stage(x, filters=256, num_blocks=2, stride=2, stage_name='stage3')
    
    # Stage 4: 512 filters, 2 blocks
    x = resnet_stage(x, filters=512, num_blocks=2, stride=2, stage_name='stage4')
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet18')
    return model

# Build the model
model = build_resnet18(input_shape=(28, 28, 3), num_classes=10)
model.summary()

# ===============================
# 5. Compile Model
# ===============================

# Adam optimizer with adaptive learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Callbacks
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
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
# 6. Data Augmentation
# ===============================
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

batch_size = 64

train_gen = train_datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)
val_gen = train_datagen.flow(x_val, y_val, batch_size=batch_size, shuffle=False)

# ===============================
# 7. Train Model
# ===============================
print("\nTraining ResNet18...")
history = model.fit(
    train_gen,
    epochs=30,
    steps_per_epoch=len(x_train) // batch_size,
    validation_data=val_gen,
    validation_steps=len(x_val) // batch_size,
    callbacks=[lr_scheduler, early_stop]
)

# ===============================
# 8. Evaluate on Test Set
# ===============================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# ===============================
# 9. Detailed Metrics
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
plt.title("ResNet18 Confusion Matrix - Fashion-MNIST")
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
df_report.to_csv("results/resnet18_fashion_mnist_report.csv", index=True)

# ===============================
# 10. Save Model and Visualize
# ===============================
os.makedirs("models", exist_ok=True)
model.save("models/resnet18_fashion_mnist.h5")

# Training curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('ResNet18 Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('ResNet18 Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("models/resnet18_training_curves.png")
plt.show()

# Sample predictions
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
plt.savefig("models/resnet18_sample_predictions.png")
plt.show()

print("\n==============================")
print(" ResNet18 Training Complete!")
print(f" Test Accuracy: {test_acc:.4f}")
print(f" Model Parameters: {model.count_params():,}")
print("==============================")