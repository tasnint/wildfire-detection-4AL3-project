import os
import numpy
import pandas as pd
from PIL import Image

"""
PLEASE FILL IN THE FILE DIRECTORIES BEFORE RUNNING THIS CODE WITH THE APPROPRIATE IMAGES
"""
# Data Sets
classification_sets = ["training_set/", "validation_set/", "testing_set/"]

# Classifications
classifications = ["nofire", "fire"]

dataset_directories = []

# Checking for correct PATHING and appending to file directories
for set in classification_sets:
    for classification in classifications:
        if not((os.path.isdir(f"{set}{classification}") and os.listdir(f"{set}{classification}"))):
            print(f"PLEASE FIX {set}{classification}")
        else:
            dataset_directories.append(f"{set}{classification}")
            

resized_output_directory = "_resized_images"
target_pixel_size = (100, 100) # THIS CAN BE MANIPULATED/PLAYED AROUND WITH

# Preprocessing
for data_set_directory in dataset_directories:
    for root, directory, files in os.walk(data_set_directory):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(root, file)
                
                # Create relative path structure
                rel_path = os.path.relpath(root, data_set_directory)
                output_subdir = os.path.join(f"{data_set_directory}{resized_output_directory}", rel_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                # Resize and save
                with Image.open(input_path) as img:
                    img = img.resize(target_pixel_size, Image.Resampling.LANCZOS)
                    output_path = os.path.join(output_subdir, file)
                    img.save(output_path)



# NOTE: I don't know if this is a correct model name, you want to name it into the appropriate model, then go right ahead.
class NeuralNetworkModel():

    def __init__(self, path):
        pass