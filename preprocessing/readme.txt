# ============================================
# Wildfire Detection Project – Preprocessing Experiments
# ============================================

## Purpose
This folder contains different individual preprocessing pipelines used to prepare wildfire images
for CNN training and experimentation. 
The goal is to systematically compare the effect of
different input resolutions and augmentation strategies on model accuracy,
training time, and overall computational efficiency.

By isolating preprocessing into individual files, we can easily modify and test
different setups without changing the main model training script.

---

## Original Dataset Information

Dataset:
The Wildfire Dataset (El-Madafri et3390/f14091697 al., 2023)
https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset

Resolution Insights:
- **Average:** 4057 × 3155 pixels  
- **Minimum:** 153 × 206 pixels  
- **Maximum:** 19699 × 8974 pixels  


---

## Purpose of resizing

Feeding all images at their original resolutions would be computationally infeasible.
CNNs require a fixed input shape, so all images must be **resized** to a standard size
before training. Resizing ensures:

1. Consistent tensor dimensions (e.g. 224×224×3)
2. Manageable memory usage and faster training
3. Comparable results across experiments

However, choosing the *right* resolution involves a trade-off between:

- **Detail preservation** (higher resolution → more features)
- **Computation time** (higher resolution → slower training)

So we will be experimenting with different pixel sizes!! :D
The following pixel sizes will be experimented with: 128*128*3, 299*299*3, 224*224*3, 1000*1000*3
Sources for choosing these sizes:  
https://massedcompute.com/faq-answers/?question=What%20is%20the%20optimal%20image%20size%20for%20training%20a%20CNN%20model%20for%20image%20recognition?
https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/tools/process_bbox.ipynb

Each Python file in this folder defines a data-loading and preprocessing pipeline for
a specific input resolution (and optionally different augmentation intensities).

