# preprocessing/preprocessing.py
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers


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

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
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
    """
    Runs experiments over multiple resolutions and augmentation sets to
    identify the best preprocessing configuration before final training.
    This explores combinations of (image_size x augmentation_type).
    """
    # If pixel sizes are not provided, we test multiple resolutions including very large images.
    # NOTE: 1000x1000 is extremely computationally expensive and may cause OOM errors if
    # batch size is not reduced (we handle this below with a dynamic batch-size rule).
    if pixel_sizes is None:
        pixel_sizes = [(128,128),(224,224)]
        # pixel_sizes = [(128,128)]

    # Multiple augmentation configurations:
    # Standard → balanced transforms
    # Stronger → heavier zoom + rotation
    # Minimal → effectively no augmentation
    # This allows the experiment to measure how augmentation strength interacts with resolution.
    if augmentations is None:
        augmentations = [
            {"rotation_range":15, "width_shift_range":0.1, "height_shift_range":0.1,"zoom_range":0.1, "horizontal_flip":True, "brightness_range":[0.8,1.2]},  # Standard
            {"rotation_range":0, "zoom_range":0, "horizontal_flip":False}              # Minimal
        ]
        # augmentations = [
        #     {"rotation_range":15, "width_shift_range":0.1, "height_shift_range":0.1,
        #      "zoom_range":0.1, "horizontal_flip":True, "brightness_range":[0.8,1.2]},  # Standard
        # ]

    results = [] # List of (resolution, augmentation, val_acc, avg_time)
    best_config = None # Stores the best (size, augmentation, accuracy)
    best_val_acc = 0 # Global best accuracy found across all combinations
    prev_best_val_acc = 0 # Tracks best accuracy from previous resolution for early stopping


    # ================================
    # OUTER LOOP → iterate over resolutions
    # ================================
    for size in pixel_sizes:
        print(f"\nTesting resolution: {size}")
        size_best_val = 0
        size_best_aug = None

        # ================================
        # INNER LOOP → iterate over augmentation configurations
        # ================================
        for aug in augmentations:
            print(f"   → Trying augmentation: {aug}")

            train_datagen = ImageDataGenerator(rescale=1./255, **aug)
            val_datagen = ImageDataGenerator(rescale=1./255)


            # ==============================================================
            # DYNAMIC BATCH SIZE RULE (IMPORTANT!)
            # --------------------------------------------------------------
            # Why needed?
            # Large images dramatically increase memory usage per batch.
            # A fixed batch_size=32 will crash (OOM) at 299px or 1000px.
            #
            # This rule automatically reduces batch size as resolution grows:
            # - 128px → 32 images per batch (safe)
            # - 224px → 16 images per batch (moderate memory)
            # - 299px & 1000px → 8 images per batch (prevents crashes)
            #
            # This makes the experiment stable across ALL sizes.
            # ==============================================================
            # Auto-adjust batch size for large resolutions to avoid OOM
            if size[0] >= 300:
                batch_size = 8   # Very large images must use very small batches
            elif size[0] >= 224:
                batch_size = 16 # Medium-sized images can use moderate batch sizes
            else:
                batch_size = 32  # Small images allow full batch size


            train_gen = train_datagen.flow_from_directory(
                f"{base_dir}/train", target_size=size, batch_size=batch_size, class_mode='binary')
            val_gen = val_datagen.flow_from_directory(
                f"{base_dir}/val", target_size=size, batch_size=batch_size, class_mode='binary')

            # Create CNN with input shape matching current resolution
            model = build_cnn((size[0], size[1], 3))

            start = time.time()
            # Train a quick 5-epoch model (NOT the final model — just exploratory)
            history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, verbose=1)
            end = time.time()

            # Record validation accuracy for this (resolution, augmentation) pair
            val_acc = max(history.history['val_accuracy'])
            avg_time = (end - start) / epochs
            results.append((size, aug, val_acc, avg_time))

            print(f"Val Acc={val_acc:.4f}, Time/Epoch={avg_time:.2f}s")

            # Track best augmentation for THIS resolution
            if val_acc > size_best_val:
                size_best_val = val_acc
                size_best_aug = aug

            # Track overall best configuration
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_config = (size, aug, val_acc)

        # ==============================================================
        # EARLY STOPPING LOGIC (between RESOLUTIONS)
        # --------------------------------------------------------------
        # Purpose:
        #   Avoid wasting compute on higher resolutions if they do not
        #   meaningfully outperform smaller ones (threshold = 3%).
        #
        # Why this is safe:
        #   We evaluate ALL augmentations first before comparing resolutions,
        #   so augmentation randomness won't prematurely stop the loop.
        #
        # Condition explanation:
        #   improvement = current_res_max_val_acc - previous_res_max_val_acc
        #   If this is < 0.03 (3%), we conclude larger images provide no gain.
        #
        # NOTE:
        #   This condition ONLY applies between resolutions, not augmentations.
        # ==============================================================
        improvement = size_best_val - prev_best_val_acc

        if improvement < 0.03 and size_best_val <= prev_best_val_acc:
            print(f"Early stopping: <3% improvement over previous resolution "
                f"({prev_best_val_acc:.4f} → {size_best_val:.4f}).")
            break
        # Update previous resolution's score for next iteration
        prev_best_val_acc = size_best_val


    # ================================
    # Print experiment summary
    # ================================
    print("\n=== Summary of Experiments ===")
    for r in results:
        print(f"{r[0]} → Acc={r[2]:.4f}, Time/Epoch={r[3]:.2f}s, Aug={r[1]}")

    print(f"\n Best configuration: Resolution {best_config[0]}, "
          f"Augmentation {best_config[1]}, Val Acc={best_config[2]:.4f}")

    return best_config, results
