import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
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

# Ensure image size is at least 32x32 (MobileNet minimum)
if best_size[0] < 32 or best_size[1] < 32:
    print(f"Warning: MobileNet requires minimum 32x32 input. Adjusting from {best_size} to (96, 96)")
    best_size = (96, 96)  # 96x96 is a good balance for MobileNet

# ===============================
# 3. Build MobileNetV2 Model
# ===============================

def build_mobilenet_wildfire(input_shape, num_classes=1, alpha=1.0):
    """
    Build MobileNetV2 model for wildfire detection (binary classification).
    
    MobileNetV2 advantages for wildfire detection:
    - Depthwise separable convolutions (8-9x fewer operations)
    - Inverted residual blocks with linear bottlenecks
    - Extremely lightweight: ~3.4M parameters
    - Fast inference for mobile/edge deployment
    - Perfect for real-time wildfire detection on drones, mobile devices
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes (1 for binary)
        alpha: Width multiplier (0.35, 0.5, 0.75, 1.0, 1.3, 1.4)
               Controls model size vs accuracy trade-off
    """
    # Load pre-trained MobileNetV2 (without top classification layer)
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',  # Transfer learning from ImageNet
        input_shape=input_shape,
        alpha=alpha,  # Width multiplier (1.0 = standard, 0.5 = half size)
        pooling='avg'  # Global Average Pooling
    )
    
    # Fine-tuning strategy: MobileNet is already small, so we can train more layers
    # Freeze the first 50% of layers for stability
    freeze_until = int(len(base_model.layers) * 0.5)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    print(f"MobileNetV2 (alpha={alpha}): Freezing first {freeze_until} of {len(base_model.layers)} layers")
    
    # Build complete model
    inputs = layers.Input(shape=input_shape)
    
    # MobileNet expects inputs in range [-1, 1], but we'll use [0, 1] with rescaling
    # Preprocess to MobileNet's expected range
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    
    x = base_model(x, training=False)
    
    # Custom classification head optimized for binary wildfire detection
    x = layers.Dropout(0.5, name='dropout_1')(x)
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    
    # Binary classification output (fire vs no fire)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=f'MobileNetV2_Wildfire_alpha{alpha}')
    return model

# Build the model with best image size from preprocessing
# You can adjust alpha for different size/speed trade-offs:
# alpha=0.5 -> 2M params, faster, slightly lower accuracy
# alpha=1.0 -> 3.4M params, standard, balanced
# alpha=1.4 -> 6.1M params, slower, higher accuracy
input_shape = (best_size[0], best_size[1], 3)
model = build_mobilenet_wildfire(input_shape=input_shape, num_classes=1, alpha=1.0)
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

# Adam optimizer with learning rate suitable for fine-tuning
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
print("\nTraining MobileNetV2 for Wildfire Detection...")
print("Ideal for mobile deployment: drones, smartphones, edge devices")
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
disp.plot(cmap='Oranges', values_format='d')
plt.title("MobileNetV2 Confusion Matrix - Wildfire Test Set")
plt.show()

# Compute precision, recall, F1, etc.
report = classification_report(y_true, y_pred, target_names=["No Fire", "Fire"], output_dict=True)
df_report = pd.DataFrame(report).transpose()

# Print nicely formatted report
print("\n===== Classification Report =====")
print(df_report.round(3))

# Save metrics table as CSV
os.makedirs("results", exist_ok=True)
df_report.to_csv("results/mobilenet_wildfire_classification_report.csv", index=True)

# ===============================
# 9. Save Model and Training Curves
# ===============================
os.makedirs("models", exist_ok=True)
model.save("models/mobilenet_v2_wildfire_model.h5")

# Also save in TFLite format for mobile deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("models/mobilenet_v2_wildfire_model.tflite", "wb") as f:
    f.write(tflite_model)
print("\nModel saved in both .h5 and .tflite formats for mobile deployment")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('MobileNetV2 Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('MobileNetV2 Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("models/mobilenet_wildfire_training_curves.png")
plt.show()

# ===============================
# 10. Predict on Sample Images
# ===============================
from tensorflow.keras.preprocessing import image
import time

# Show predictions on random test images + measure inference time
num_samples = 8
indices = np.random.choice(len(test_gen.filepaths), num_samples, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

inference_times = []

for i, idx in enumerate(indices):
    img_path = test_gen.filepaths[idx]
    img = image.load_img(img_path, target_size=best_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Measure inference time (important for mobile deployment)
    start_time = time.time()
    prediction = model.predict(img_array, verbose=0)[0][0]
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    inference_times.append(inference_time)
    
    true_label = "Fire" if test_gen.classes[idx] == 1 else "No Fire"
    pred_label = "Fire" if prediction > 0.5 else "No Fire"
    
    axes[i].imshow(img)
    color = 'green' if pred_label == true_label else 'red'
    axes[i].set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {prediction:.3f}\nTime: {inference_time:.1f}ms", 
                      color=color, fontsize=9)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("models/mobilenet_wildfire_sample_predictions.png")
plt.show()

avg_inference_time = np.mean(inference_times)
print(f"\nAverage inference time: {avg_inference_time:.2f}ms")

# ===============================
# 11. Model Performance Summary
# ===============================
print("\n" + "="*60)
print(" MobileNetV2 Wildfire Detection - Training Complete!")
print("="*60)
print(f" Test Accuracy: {test_acc:.4f}")
print(f" Test Loss: {test_loss:.4f}")
print(f" Total Parameters: {model.count_params():,}")
print(f" Trainable Parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
print(f" Average Inference Time: {avg_inference_time:.2f}ms")
print(f" Image Size: {best_size}")
print(f" Augmentation: {best_aug}")
print("\n MobileNetV2 Advantages for Wildfire Detection:")
print(" ✓ Depthwise separable convolutions (8-9x faster)")
print(" ✓ Only ~3.4M parameters (vs ResNet50's 25M)")
print(" ✓ Fast inference for real-time detection")
print(" ✓ Perfect for mobile deployment:")
print("   • Drone-based wildfire monitoring")
print("   • Smartphone apps for field workers")
print("   • Raspberry Pi / edge devices")
print("   • Real-time video surveillance")
print(" ✓ Saved in TFLite format for mobile apps")
print(" ✓ Transfer learning from ImageNet")
print("\n Deployment Options:")
print(" • Use .h5 model for server/desktop deployment")
print(" • Use .tflite model for Android/iOS apps")
print(" • Can quantize to INT8 for even faster inference")
print("="*60)