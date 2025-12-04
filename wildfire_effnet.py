import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
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

# Ensure image size is at least 32x32 (EfficientNet minimum)
if best_size[0] < 32 or best_size[1] < 32:
    print(f"Warning: EfficientNet requires minimum 32x32 input. Adjusting from {best_size} to (32, 32)")
    best_size = (32, 32)

# ===============================
# 3. Build EfficientNet-B0 Model
# ===============================

def build_efficientnet_wildfire(input_shape, num_classes=1):
    """
    Build EfficientNet-B0 model for wildfire detection (binary classification).
    
    EfficientNet advantages for wildfire detection:
    - Compound scaling for optimal accuracy/efficiency balance
    - MBConv blocks with depthwise separable convolutions
    - Squeeze-and-Excitation for channel attention
    - Pre-trained on ImageNet for transfer learning
    - Much more efficient than ResNet50 (5M vs 25M parameters)
    """
    # Load pre-trained EfficientNet-B0 (without top classification layer)
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',  # Transfer learning from ImageNet
        input_shape=input_shape,
        pooling='avg'  # Global Average Pooling
    )
    
    # Fine-tuning strategy: freeze early layers, train later layers
    # For large datasets like wildfire (9.9GB), we can train more layers
    # Freeze the first 60% of layers
    freeze_until = int(len(base_model.layers) * 0.6)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    print(f"EfficientNet-B0: Freezing first {freeze_until} of {len(base_model.layers)} layers")
    
    # Build complete model
    inputs = layers.Input(shape=input_shape)
    
    # EfficientNet expects inputs in range [0, 1] (handled by rescale in data generator)
    x = base_model(inputs, training=False)
    
    # Custom classification head for wildfire detection
    x = layers.Dropout(0.5, name='dropout_1')(x)
    x = layers.Dense(256, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.4, name='dropout_2')(x)
    x = layers.Dense(128, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.3, name='dropout_3')(x)
    
    # Binary classification output (fire vs no fire)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='EfficientNetB0_Wildfire')
    return model

# Build the model with best image size from preprocessing
input_shape = (best_size[0], best_size[1], 3)
model = build_efficientnet_wildfire(input_shape=input_shape, num_classes=1)
model.summary()

print(f"\nTotal parameters: {model.count_params():,}")
print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
print(f"Non-trainable parameters: {sum([tf.size(w).numpy() for w in model.non_trainable_weights]):,}")

# ===============================
# 4. Prepare Data with Best Config
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
# 5. Compile Model
# ===============================

# Lower learning rate for fine-tuning pre-trained model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Callbacks for adaptive training
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
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ===============================
# 6. Train Model
# ===============================
print("\nTraining EfficientNet-B0 for Wildfire Detection...")
history = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen,
    callbacks=[lr_scheduler, early_stop]
)

# ===============================
# 7. Evaluate on Test Set
# ===============================
test_loss, test_acc = model.evaluate(test_gen)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# ===============================
# 8. Detailed Performance Metrics
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
plt.title("EfficientNet-B0 Confusion Matrix - Wildfire Test Set")
plt.show()

# Compute precision, recall, F1, etc.
report = classification_report(y_true, y_pred, target_names=["No Fire", "Fire"], output_dict=True)
df_report = pd.DataFrame(report).transpose()

# Print nicely formatted report
print("\n===== Classification Report =====")
print(df_report.round(3))

# Save metrics table as CSV
os.makedirs("results", exist_ok=True)
df_report.to_csv("results/efficientnet_wildfire_classification_report.csv", index=True)

# ===============================
# 9. Save Model and Training Curves
# ===============================
os.makedirs("models", exist_ok=True)
model.save("models/efficientnet_b0_wildfire_model.h5")

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
plt.savefig("models/efficientnet_wildfire_training_curves.png")
plt.show()

# ===============================
# 10. Predict on Sample Images
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
plt.savefig("models/efficientnet_wildfire_sample_predictions.png")
plt.show()

# ===============================
# 11. Model Performance Summary
# ===============================
print("\n==============================")
print(" EfficientNet-B0 Wildfire Training Complete!")
print("==============================")
print(f" Test Accuracy: {test_acc:.4f}")
print(f" Test Loss: {test_loss:.4f}")
print(f" Total Parameters: {model.count_params():,}")
print(f" Trainable Parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
print(f" Image Size: {best_size}")
print(f" Augmentation: {best_aug}")
print("\n EfficientNet-B0 Advantages:")
print(" • Compound scaling (depth + width + resolution)")
print(" • MBConv blocks with Squeeze-and-Excitation")
print(" • Transfer learning from ImageNet")
print(" • ~5.3M base parameters (vs ResNet50's 25M)")
print(" • Higher accuracy with lower computational cost")
print(" • Ideal for large datasets like wildfire (9.9GB)")
print("==============================")