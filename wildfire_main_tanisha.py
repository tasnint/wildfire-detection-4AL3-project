import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing import image

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
print(f"\n Using best preprocessing config: {best_size}, augmentation={best_aug}")

# ===============================
# 3. Prepare Data with Best Config
# ===============================
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, **best_aug)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=best_size, batch_size=32, class_mode='binary')
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=best_size, batch_size=32, class_mode='binary')
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=best_size, batch_size=32, class_mode='binary', shuffle=False)

# ===============================
# 4. CNN Model Architecture
# ===============================
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(best_size[0], best_size[1], 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),    
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # efficient net
    # resnet
    # mobilenet

    # layers.Flatten(),
    # layers.Dense(256, activation='relu'),
    # layers.Dropout(0.5),
    # layers.Dense(1, activation='sigmoid')

    # ============================================
    #  REPLACEMENT: Global Average Pooling layer
    # --------------------------------------------
    # GAP drastically reduces parameters compared to Flatten.
    # Instead of flattening the entire feature map (which produces tens/hundreds 
    # of thousands of values), GAP averages each feature map into a single number.
    #
    # Benefits:
    #   • Less overfitting
    #   • Fewer parameters → faster training
    #   • Better generalization (especially for large datasets like ours which is 9.9GB)
    #   • Mimics behavior of modern CNNs (ResNet, EfficientNet)
    # ============================================
    layers.GlobalAveragePooling2D(),

    # Smaller Dense layer to reduce parameters and overfitting risk
    # --------------------------------------------------------------
    # Why Dense(128) instead of Dense(256) after GlobalAveragePooling?
    # --------------------------------------------------------------
    # Before using GAP, the model used Flatten(), which produced a VERY 
    # large vector (e.g., 8×8×128 = 8192 values). Feeding this into a 
    # Dense(256) layer required over 2 million parameters, which is large,
    # slow to train, and prone to overfitting.
    #
    # After switching to GlobalAveragePooling2D(), the feature map is 
    # reduced to just 128 values (one per feature channel). Because the
    # input vector is now much smaller, we no longer need a large 256-unit 
    # dense layer. A Dense(128) layer is a natural fit:
    #
    #    - fewer parameters (16k instead of 2M+) → prevents overfitting
    #    - faster training and inference
    #    - matches modern CNN architecture patterns (ResNet/MobileNet)
    #
    # In short: GAP dramatically shrinks the feature vector, so reducing
    # the dense layer size to 128 keeps the model efficient, stable, and 
    # better-regularized.
# --------------------------------------------------------------

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),

    # Final output neuron (binary classification)
    layers.Dense(1, activation='sigmoid')

])

# ============================================
#  OPTIMIZER + CALLBACKS (Adaptive Learning Rate)
# ============================================

# Adam optimizer with automatic adaptive learning rate.
# (We do NOT manually specify a learning_rate, so Adam
#  uses its built-in adaptive LR mechanism.)
optimizer = tf.keras.optimizers.Adam()
# Automatically reduce the learning rate when validation loss stops improving.
# This helps stabilize training and escape plateaus.
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',     # watch validation loss
    factor=0.5,             # reduce LR by 50%
    patience=3,             # wait 3 epochs of no improvement
    min_lr=1e-6             # do not let LR fall below this
)

# Stop training when validation loss stops improving
# and restore the best model weights.
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


# Compile the model with binary crossentropy loss (binary classification)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ===============================
# 5. Train Final Model
# ===============================
# ============================================
#  TRAINING LOOP WITH CALLBACKS
# ============================================
# The callbacks enable:
#   - automatic LR tuning (ReduceLROnPlateau)
#   - preventing overfitting (EarlyStopping)
history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[lr_scheduler, early_stop]
)

# ===============================
# 6. Evaluate on Test Set
# ===============================
test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc:.4f}")

# ===============================
# 6.1 Detailed Performance Metrics
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
plt.title("Confusion Matrix - Test Set")
plt.show()

# Compute precision, recall, F1, etc.
report = classification_report(y_true, y_pred, target_names=["No Fire", "Fire"], output_dict=True)
df_report = pd.DataFrame(report).transpose()

# Print nicely formatted report
print("\n===== Classification Report =====")
print(df_report.round(3))

# Save metrics table as CSV
os.makedirs("results", exist_ok=True)
df_report.to_csv("results/test_classification_report.csv", index=True)

# ===============================
# 7. Save Model and Training Curves
# ===============================
os.makedirs("models", exist_ok=True)
model.save("models/wildfire_cnn_best_model.h5")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("models/training_curves_best.png")
plt.show()

# ===============================
# 8. Predict on a Sample Image
# ===============================
sample_img_path = test_gen.filepaths[0]
img = image.load_img(sample_img_path, target_size=best_size)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
label = "Fire" if prediction > 0.5 else "No Fire"
print(f"Predicted: {label} ({prediction:.3f}) for image {os.path.basename(sample_img_path)}")

print("\n==============================")
print(" Final training configuration:")
print(f"Best image size: {best_size}")
print(f"Best augmentation: {best_aug}")
print(f"Validation accuracy during search: {best_acc:.4f}")
print("==============================\n")