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
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ===============================
# 5. Train Final Model
# ===============================
history = model.fit(train_gen, epochs=20, validation_data=val_gen)

# ===============================
# 6. Evaluate on Test Set
# ===============================
test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc:.4f}")

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
