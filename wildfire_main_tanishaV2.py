import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import pandas as pd
from tensorflow.keras.preprocessing import image

# Import preprocessing experiment logic
from preprocessing.preprocessing import run_preprocessing_experiments

# Setup
base_dir = "data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ===============================
# 1. DATA DESCRIPTION & CLASS DISTRIBUTION
# ===============================
def analyze_dataset_distribution(base_dir):
    print("\n" + "="*80)
    print(" DATASET ANALYSIS - Class Distribution")
    print("="*80)
    
    splits = ['train', 'val', 'test']
    total_stats = {}
    
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        fire_count = len(os.listdir(os.path.join(split_dir, 'fire'))) if os.path.exists(os.path.join(split_dir, 'fire')) else 0
        nofire_count = len(os.listdir(os.path.join(split_dir, 'nofire'))) if os.path.exists(os.path.join(split_dir, 'nofire')) else 0
        total = fire_count + nofire_count
        
        print(f"\n{split.upper()} SET:")
        print(f"  Fire images:    {fire_count:,} ({fire_count/total*100:.1f}%)")
        print(f"  No-Fire images: {nofire_count:,} ({nofire_count/total*100:.1f}%)")
        print(f"  Total:          {total:,}")
        print(f"  Class ratio:    {fire_count/nofire_count:.3f}" if nofire_count > 0 else "")
        
        total_stats[split] = {'fire': fire_count, 'nofire': nofire_count, 'total': total}
    
    total_fire = sum([total_stats[s]['fire'] for s in splits])
    total_nofire = sum([total_stats[s]['nofire'] for s in splits])
    grand_total = total_fire + total_nofire
    
    print(f"\nOVERALL DATASET:")
    print(f"  Total Fire:    {total_fire:,} ({total_fire/grand_total*100:.1f}%)")
    print(f"  Total No-Fire: {total_nofire:,} ({total_nofire/grand_total*100:.1f}%)")
    print(f"  Grand Total:   {grand_total:,}")
    print(f"  Balance:       {'BALANCED' if 0.4 < total_fire/grand_total < 0.6 else 'IMBALANCED'}")
    print("="*80 + "\n")
    
    return total_stats

dataset_stats = analyze_dataset_distribution(base_dir)

# ===============================
# 2. RUN PREPROCESSING EXPERIMENTS
# ===============================
print("Starting preprocessing & augmentation experiments...")
best_config, results = run_preprocessing_experiments(base_dir, epochs=5)
best_size, best_aug, best_acc = best_config
print(f"\nUsing best preprocessing config: {best_size}, augmentation={best_aug}")

# ===============================
# 3. PREPARE DATA WITH BEST CONFIG
# ===============================
BATCH_SIZE = 64

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, **best_aug)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=best_size, batch_size=BATCH_SIZE, class_mode='binary')
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=best_size, batch_size=BATCH_SIZE, class_mode='binary')
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=best_size, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

print(f"\nBatch size: {BATCH_SIZE}")

# ===============================
# 4. BUILD CNN MODEL WITH ARCHITECTURAL DETAILS
# ===============================
def build_cnn_wildfire(input_shape, num_classes=1):
    print("\n" + "="*80)
    print(" BUILDING CUSTOM CNN MODEL")
    print("="*80)
    
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
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
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    print(f"\nArchitecture:")
    print(f"  Layer 1: Conv2D(16, 3x3) -> BN -> MaxPool(2x2)")
    print(f"  Layer 2: Conv2D(32, 3x3) -> BN -> MaxPool(2x2)")
    print(f"  Layer 3: Conv2D(64, 3x3) -> BN -> MaxPool(2x2)")
    print(f"  Layer 4: Conv2D(128, 3x3) -> BN -> MaxPool(2x2) -> Dropout(0.3)")
    print(f"  Head: GlobalAvgPool -> Dense(128) -> Dropout(0.4) -> Dense(1, Sigmoid)")
    print(f"  Regularization: BatchNorm + Dropout(0.3, 0.4)")
    print("="*80 + "\n")
    
    return model

input_shape = (best_size[0], best_size[1], 3)
model = build_cnn_wildfire(input_shape, 1)
model.summary()

print(f"\nTotal params: {model.count_params():,}\n")

# ===============================
# 5. COMPILE MODEL
# ===============================
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

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
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(name='auc')]
)

# ===============================
# 6. TRAIN MODEL
# ===============================
print("Training Custom CNN for Wildfire Detection...")
history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[lr_scheduler, early_stop]
)

# ===============================
# 7. EVALUATION WITH ERROR ANALYSIS
# ===============================
print("\n" + "="*80)
print(" TEST SET EVALUATION & ERROR ANALYSIS")
print("="*80)

test_results = model.evaluate(test_gen)
test_loss, test_acc, test_precision, test_recall, test_auc = test_results

print(f"\nTest Results:")
print(f"  Accuracy:  {test_acc:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  AUC:       {test_auc:.4f}")
print(f"  Loss:      {test_loss:.4f}\n")

y_true = test_gen.classes
y_pred_probs = model.predict(test_gen).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:")
print(f"  True Negatives (TN):  {tn:,} - Correct No-Fire")
print(f"  False Positives (FP): {fp:,} - False Alarms")
print(f"  False Negatives (FN): {fn:,} - Missed Fires (CRITICAL)")
print(f"  True Positives (TP):  {tp:,} - Correct Fire")

fpr_rate = fp/(fp+tn)*100 if (fp+tn) > 0 else 0
fnr_rate = fn/(fn+tp)*100 if (fn+tp) > 0 else 0

print(f"\nError Rates:")
print(f"  FPR: {fpr_rate:.2f}% - False alarms")
print(f"  FNR: {fnr_rate:.2f}% - Missed fires (CRITICAL)")
print(f"  Risk: {'🔴 HIGH' if fnr_rate > 10 else '🟡 MEDIUM' if fnr_rate > 5 else '🟢 LOW'}")

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fire", "Fire"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Custom CNN")
plt.tight_layout()
plt.savefig("models/cnn_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

report = classification_report(y_true, y_pred, target_names=["No Fire", "Fire"], output_dict=True)
df_report = pd.DataFrame(report).transpose()
print(f"\nClassification Report:\n{df_report.round(3)}")
df_report.to_csv("results/cnn_classification_report.csv")

# ===============================
# 8. ROC/AUC ANALYSIS
# ===============================
print("\n" + "="*80)
print(" ROC/AUC & PRECISION-RECALL ANALYSIS")
print("="*80)

fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = roc_auc_score(y_true, y_pred_probs)
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_probs)
avg_precision = average_precision_score(y_true, y_pred_probs)

print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}\n")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(fpr, tpr, 'darkorange', lw=2, label=f'ROC (AUC={roc_auc:.3f})')
axes[0].plot([0,1], [0,1], 'navy', lw=2, linestyle='--', label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(recall_curve, precision_curve, 'blue', lw=2, label=f'PR (AP={avg_precision:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(thresholds[:-1], fpr[:-1], 'red', lw=2, label='FPR')
axes[2].plot(thresholds[:-1], tpr[:-1], 'green', lw=2, label='TPR')
axes[2].axvline(x=0.5, color='black', linestyle='--', label='Default (0.5)')
axes[2].set_xlabel('Threshold')
axes[2].set_ylabel('Rate')
axes[2].set_title('Threshold Analysis')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("models/cnn_roc_pr_threshold_analysis.png", dpi=150, bbox_inches='tight')
plt.close()

# ===============================
# 9. STRATEGY FOR REDUCING FALSE NEGATIVES
# ===============================
print("\n" + "="*80)
print(" STRATEGY FOR REDUCING FALSE NEGATIVES")
print("="*80)

print("\nThreshold Optimization:")
print("─"*80)
print(f"{'Threshold':<12} {'FN':<8} {'FN Rate':<12} {'Recall':<10} {'Precision':<12}")
print("─"*80)

test_thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
threshold_results = []

for thresh in test_thresholds:
    y_pred_thresh = (y_pred_probs > thresh).astype(int)
    cm_thresh = confusion_matrix(y_true, y_pred_thresh)
    tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel()
    
    recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    fn_rate_t = fn_t / (fn_t + tp_t) * 100 if (fn_t + tp_t) > 0 else 0
    
    threshold_results.append({
        'threshold': thresh,
        'fn': fn_t,
        'recall': recall_t,
        'precision': precision_t,
        'fn_rate': fn_rate_t
    })
    
    print(f"{thresh:<12.2f} {fn_t:<8} {fn_rate_t:<12.2f} {recall_t:<10.3f} {precision_t:<12.3f}")

best_thresh = min(threshold_results, key=lambda x: x['fn_rate'])
print(f"\n✅ RECOMMENDED THRESHOLD: {best_thresh['threshold']:.2f}")
print(f"   Reduces FN to {best_thresh['fn']} ({best_thresh['fn_rate']:.2f}%)")
print(f"   Recall: {best_thresh['recall']:.3f}, Precision: {best_thresh['precision']:.3f}")

print("\n" + "="*80)
print(" ACTIONABLE NEXT STEPS")
print("="*80)
print("""
1. THRESHOLD TUNING (Immediate):
   • Lower threshold to 0.3-0.4 to reduce missed fires
   • Trade-off: More false alarms, fewer disasters

2. CLASS IMBALANCE (Short-term):
   • Apply class weights: {0: 1.0, 1: 2.0}
   • Use focal loss for hard examples

3. ARCHITECTURE IMPROVEMENTS (Medium-term):
   • Try pretrained models (EfficientNet, MobileNet, ResNet)
   • Add attention mechanisms
   • Deeper networks with residual connections

4. DATA AUGMENTATION (Medium-term):
   • Oversample fire images
   • Fire-specific augmentations (brightness, smoke effects)
   • Collect diverse wildfire imagery

5. ENSEMBLE METHODS (Medium-term):
   • Combine CNN + EfficientNet + MobileNet
   • Voting/averaging reduces model-specific errors

6. DEPLOYMENT (Production):
   • Use threshold 0.3-0.4
   • Two-stage: sensitive screening + confirmation
   • Human-in-the-loop for borderline predictions
   • Real-time alerting with confidence scores

7. MONITORING (Continuous):
   • Track FN rate in production
   • Collect edge cases for retraining
   • Quarterly model updates
   • A/B test threshold optimization
""")

# ===============================
# 10. SAVE MODEL & TRAINING CURVES
# ===============================
model.save("models/wildfire_cnn_best_model.h5")
print("\n✅ Model saved to models/wildfire_cnn_best_model.h5")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0,0].plot(history.history['accuracy'], label='train')
axes[0,0].plot(history.history['val_accuracy'], label='val')
axes[0,0].set_title('Accuracy')
axes[0,0].set_xlabel('Epoch')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

axes[0,1].plot(history.history['loss'], label='train')
axes[0,1].plot(history.history['val_loss'], label='val')
axes[0,1].set_title('Loss')
axes[0,1].set_xlabel('Epoch')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

axes[0,2].plot(history.history['precision'], label='train')
axes[0,2].plot(history.history['val_precision'], label='val')
axes[0,2].set_title('Precision')
axes[0,2].set_xlabel('Epoch')
axes[0,2].legend()
axes[0,2].grid(alpha=0.3)

axes[1,0].plot(history.history['recall'], label='train')
axes[1,0].plot(history.history['val_recall'], label='val')
axes[1,0].set_title('Recall')
axes[1,0].set_xlabel('Epoch')
axes[1,0].legend()
axes[1,0].grid(alpha=0.3)

axes[1,1].plot(history.history['auc'], label='train')
axes[1,1].plot(history.history['val_auc'], label='val')
axes[1,1].set_title('AUC')
axes[1,1].set_xlabel('Epoch')
axes[1,1].legend()
axes[1,1].grid(alpha=0.3)

if 'lr' in history.history:
    axes[1,2].plot(history.history['lr'])
    axes[1,2].set_title('Learning Rate')
    axes[1,2].set_xlabel('Epoch')
    axes[1,2].set_yscale('log')
    axes[1,2].grid(alpha=0.3)
else:
    axes[1,2].axis('off')

plt.tight_layout()
plt.savefig("models/training_curves_best.png", dpi=150, bbox_inches='tight')
plt.close()

# ===============================
# 11. SAMPLE PREDICTIONS
# ===============================
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
    pred_optimized = "Fire" if prediction > best_thresh['threshold'] else "No Fire"
    
    axes[i].imshow(img)
    color = 'green' if pred_label == true_label else 'red'
    axes[i].set_title(f"True: {true_label}\nPred(0.5): {pred_label}\nPred({best_thresh['threshold']:.2f}): {pred_optimized}\nConf: {prediction:.3f}", 
                      color=color, fontsize=9)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("models/cnn_sample_predictions.png", dpi=150, bbox_inches='tight')
plt.close()

# ===============================
# SUMMARY
# ===============================
print("\n" + "="*80)
print(" TRAINING COMPLETE - SUMMARY")
print("="*80)
print(f" Test Accuracy:  {test_acc:.4f}")
print(f" Test Precision: {test_precision:.4f}")
print(f" Test Recall:    {test_recall:.4f}")
print(f" Test AUC:       {roc_auc:.4f}")
print(f" False Negatives: {fn} ({fnr_rate:.2f}%)")
print(f" Recommended Threshold: {best_thresh['threshold']:.2f}")
print(f" Total Parameters: {model.count_params():,}")
print(f" Best image size: {best_size}")
print(f" Best augmentation: {best_aug}")
print(f" Batch size: {BATCH_SIZE}")
print(f" Learning rate: 0.001")
print("="*80)