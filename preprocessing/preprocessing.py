# preprocessing/preprocessing.py
import time
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks


def augment_image(image, label):
    """Apply data augmentation to training images."""
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def build_cnn(input_shape):
    """Builds a simple CNN for binary classification (Fire / No Fire)."""
    model = models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Dropout(0.3),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def run_preprocessing_experiments(base_dir, pixel_sizes=None, augmentations=None, epochs=3):
    """Runs experiments over multiple resolutions and augmentation sets using tf.data."""
    if pixel_sizes is None:
        pixel_sizes = [(128,128), (224,224), (299,299), (1000,1000)]  # Can add (224,224) if needed

    if augmentations is None:
        # We'll test with and without augmentation
        augmentations = [True, False]  # True = with augmentation, False = without

    results = []
    best_config = None
    best_val_acc = 0
    prev_best_val_acc = 0
    
    BATCH_SIZE = 64
    AUTOTUNE = tf.data.AUTOTUNE

    # Outer loop → resolutions
    for size in pixel_sizes:
        print(f"\nTesting resolution: {size}")
        size_best_val = 0
        size_best_aug = None

        # Inner loop → augmentations
        for use_aug in augmentations:
            aug_str = "with augmentation" if use_aug else "without augmentation"
            print(f"   → Trying {aug_str}")

            # Load datasets
            train_ds = tf.keras.utils.image_dataset_from_directory(
                f"{base_dir}/train",
                image_size=size,
                batch_size=BATCH_SIZE,
                label_mode='binary',
                shuffle=True,
                seed=123
            )
            
            val_ds = tf.keras.utils.image_dataset_from_directory(
                f"{base_dir}/val",
                image_size=size,
                batch_size=BATCH_SIZE,
                label_mode='binary',
                shuffle=False,
                seed=123
            )

            # Normalize
            normalization_layer = layers.Rescaling(1./255)
            train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), 
                                   num_parallel_calls=AUTOTUNE)
            val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), 
                               num_parallel_calls=AUTOTUNE)

            # Apply augmentation if needed
            if use_aug:
                train_ds = train_ds.map(augment_image, num_parallel_calls=AUTOTUNE)

            # Optimize pipeline
            train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

            model = build_cnn((size[0], size[1], 3))

            # Add early stopping for efficiency
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True,
                verbose=0
            )

            start = time.time()
            history = model.fit(
                train_ds, 
                epochs=epochs, 
                validation_data=val_ds, 
                verbose=0,
                callbacks=[early_stop]
            )
            end = time.time()

            val_acc = max(history.history['val_accuracy'])
            avg_time = (end - start) / len(history.history['val_accuracy'])
            results.append((size, aug_str, val_acc, avg_time))

            print(f"Val Acc={val_acc:.4f}, Time/Epoch={avg_time:.2f}s")

            if val_acc > size_best_val:
                size_best_val = val_acc
                size_best_aug = use_aug

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_config = (size, use_aug, val_acc)

        # Early stop between pixel sizes (Δacc < 0.03)
        if (size_best_val - prev_best_val_acc) < 0.03 and size_best_val > prev_best_val_acc:
            print(f"Early stopping: improvement < 0.03 "
                  f"({prev_best_val_acc:.4f} → {size_best_val:.4f})")
            break

        prev_best_val_acc = size_best_val

    print("\n=== Summary of Experiments ===")
    for r in results:
        print(f"{r[0]} → Acc={r[2]:.4f}, Time/Epoch={r[3]:.2f}s, Aug={r[1]}")

    print(f"\nBest configuration: Resolution {best_config[0]}, "
          f"Augmentation={'enabled' if best_config[1] else 'disabled'}, "
          f"Val Acc={best_config[2]:.4f}")

    return best_config, results