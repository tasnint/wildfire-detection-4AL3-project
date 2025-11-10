# preprocessing/preprocessing.py
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers


def build_cnn(input_shape):
    """Builds a simple CNN for binary classification (Fire / No Fire)."""
    model = models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),

        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def run_preprocessing_experiments(base_dir, pixel_sizes=None, augmentations=None, epochs=5):
    """Runs experiments over multiple resolutions and augmentation sets."""
    if pixel_sizes is None:
        # pixel_sizes = [(128,128), (224,224), (299,299), (1000,1000)]
        pixel_sizes = [(128,128)]

    if augmentations is None:
        # augmentations = [
        #     {"rotation_range":15, "width_shift_range":0.1, "height_shift_range":0.1,
        #      "zoom_range":0.1, "horizontal_flip":True, "brightness_range":[0.8,1.2]},  # Standard
        #     {"rotation_range":30, "zoom_range":0.2, "horizontal_flip":True},            # Stronger
        #     {"rotation_range":0, "zoom_range":0, "horizontal_flip":False},              # Minimal
        # ]
        augmentations = [
            {"rotation_range":15, "width_shift_range":0.1, "height_shift_range":0.1,
             "zoom_range":0.1, "horizontal_flip":True, "brightness_range":[0.8,1.2]},  # Standard
        ]

    results = []
    best_config = None
    best_val_acc = 0
    prev_best_val_acc = 0

    # Outer loop → resolutions
    for size in pixel_sizes:
        print(f"\nTesting resolution: {size}")
        size_best_val = 0
        size_best_aug = None

        # Inner loop → augmentations
        for aug in augmentations:
            print(f"   → Trying augmentation: {aug}")

            train_datagen = ImageDataGenerator(rescale=1./255, **aug)
            val_datagen = ImageDataGenerator(rescale=1./255)

            train_gen = train_datagen.flow_from_directory(
                f"{base_dir}/train", target_size=size, batch_size=32, class_mode='binary')
            val_gen = val_datagen.flow_from_directory(
                f"{base_dir}/val", target_size=size, batch_size=32, class_mode='binary')

            model = build_cnn((size[0], size[1], 3))

            start = time.time()
            history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, verbose=1)
            end = time.time()

            val_acc = max(history.history['val_accuracy'])
            avg_time = (end - start) / epochs
            results.append((size, aug, val_acc, avg_time))

            print(f"Val Acc={val_acc:.4f}, Time/Epoch={avg_time:.2f}s")

            if val_acc > size_best_val:
                size_best_val = val_acc
                size_best_aug = aug

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_config = (size, aug, val_acc)

        # Early stop between pixel sizes (Δacc < 0.03)
        if (size_best_val - prev_best_val_acc) < 0.03 and size_best_val > prev_best_val_acc:
            print(f"Early stopping: improvement < 0.03 "
                  f"({prev_best_val_acc:.4f} → {size_best_val:.4f})")
            break

        prev_best_val_acc = size_best_val

    print("\n=== Summary of Experiments ===")
    for r in results:
        print(f"{r[0]} → Acc={r[2]:.4f}, Time/Epoch={r[3]:.2f}s, Aug={r[1]}")

    print(f"\n Best configuration: Resolution {best_config[0]}, "
          f"Augmentation {best_config[1]}, Val Acc={best_config[2]:.4f}")

    return best_config, results
