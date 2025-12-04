import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# Import preprocessing experiment logic
from preprocessing.preprocessing import run_preprocessing_experiments

# ===============================
# 1. Directory Paths
# ===============================
base_dir = "data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# ===============================
# 2. Run Preprocessing Experiments
# ===============================
print("Starting preprocessing & augmentation experiments...")
best_config, results = run_preprocessing_experiments(base_dir, epochs=5)
best_size, best_aug, best_acc = best_config
print(f"\nUsing best preprocessing config: {best_size}, augmentation={best_aug}")

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
    
    # Add skip connection (residual connection)
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

def build_resnet18(input_shape, num_classes=1):
    """
    Build ResNet18 architecture for wildfire detection:
    - Initial conv layer with 7x7 kernel
    - MaxPooling for initial downsampling
    - 4 stages with [2, 2, 2, 2] blocks
    - Global Average Pooling
    - Dense output layer for binary classification
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution (7x7 kernel, standard ResNet)
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same', name='maxpool1')(x)
    
    # ResNet18 has 4 stages with [2, 2, 2, 2] blocks
    # Stage 1: 64 filters, 2 blocks
    x = resnet_stage(x, filters=64, num_blocks=2, stride=1, stage_name='stage1')
    
    # Stage 2: 128 filters, 2 blocks
    x = resnet_stage(x, filters=128, num_blocks=2, stride=2, stage_name='stage2')
    
    # Stage 3: 256 filters, 2 blocks
    x = resnet_stage(x, filters=256, num_blocks=2, stride=2, stage_name='stage3')
    
    # Stage 4: 512 filters, 2 blocks
    x = resnet_stage(x, filters=512, num_blocks=2, stride=2, stage_name='stage4')
    
    # Global Average Pooling (reduces overfitting)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dropout for regularization
    x = layers.Dropout(0.5, name='dropout')(x)
    
    # Output layer for binary classification (fire vs no fire)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet18_Wildfire')
    return model

# Build the model with best image size from preprocessing
input_shape = (best_size[0], best_size[1], 3)
model = build_resnet18(input_shape=input_shape, num_classes=1)
model.summary()

# ===============================
# 5. Prepare Data with Best Config
# ===============================
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    **best_aug
)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=best_size, batch_size=32, class_mode='binary')
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=best_size, batch_size=32, class_mode='binary')
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=best_size, batch_size=32, class_mode='binary', shuffle=False)

# ===============================
# 6. Compile Model
# ===============================

# Adam optimizer with adaptive learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Callbacks for adaptive training
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
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ===============================
# 7. Train Model
# ===============================
print("\nTraining ResNet18 for Wildfire Detection...")
history = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen,
    callbacks=[lr_scheduler, early_stop]
)

# ===============================
# 8. Evaluate on Test Set
# ===============================
test_loss, test_acc = model.evaluate(test_gen)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# ===============================
# 9. Detailed Performance Metrics
# ===============================
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd

# Get true labels and predictions
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fire", "Fire"])
disp.plot(cmap='Blues', values_format='d')
plt.title("ResNet18 Confusion Matrix - Wildfire Test Set")
plt.show()

# Compute precision, recall, F1, etc.
report = classification_report(y_true, y_pred, target_names=["No Fire", "Fire"], output_dict=True)
df_report = pd.DataFrame(report).transpose()

# Print nicely formatted report
print("\n===== Classification Report =====")
print(df_report.round(3))

# Save metrics table as CSV
os.makedirs("results", exist_ok=True)
df_report.to_csv("results/resnet18_wildfire_classification_report.csv", index=True)

# ===============================
# 10. Save Model and Training Curves
# ===============================
os.makedirs("models", exist_ok=True)
model.save("models/resnet18_wildfire_model.h5")

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
plt.savefig("models/resnet18_wildfire_training_curves.png")
plt.show()

# ===============================
# 11. Predict on Sample Images
# ===============================
from tensorflow.keras.preprocessing import image

# Show predictions on random test images
num_samples = 8
indices = np.random.choice(len(test_gen.filepaths), num_samples, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for i, idx in enumerate(indices):
    img_path = test_gen.filepaths[idx]
    img = image.load_img(img_path, target_size=best_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)[0][0]
    true_label = "Fire" if test_gen.classes[idx] == 1 else "No Fire"
    pred_label = "Fire" if prediction > 0.5 else "No Fire"
    
    axes[i].imshow(img)
    color = 'green' if pred_label == true_label else 'red'
    axes[i].set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {prediction:.3f}", 
                      color=color, fontsize=10)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("models/resnet18_wildfire_sample_predictions.png")
plt.show()

print("\n==============================")
print(" ResNet18 Wildfire Training Complete!")
print(f" Model Parameters: {model.count_params():,}")
print(f" Test Accuracy: {test_acc:.4f}")
print(f" Best image size used: {best_size}")
print(f" Best augmentation config: {best_aug}")
print("==============================")