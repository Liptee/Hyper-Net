import numpy as np

from config import *
from utils import load_data
from utils import create_binary_masks, extract_array_from_matfile
from scripts.utils import get_device
from scripts.train_model import train_model

CUDA_DEVICE = get_device(cuda_device)

create_binary_masks(path_to_mask, index_in_mask_file)
all_files_binary_masks = load_data("checkpoints/bi_masks", "mat")
img = extract_array_from_matfile(path_to_img, "image")
print(f"iamge shape: {img.shape}")

hyperparams = {
    "epochs": epochs,
    "device": CUDA_DEVICE,
    "patch_size": patch_size,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "n_bands": img.shape[-1],
    "ignored_labels":[0]
}

cnn_params = {
    #conv1
    "c1_core_counts": 16,
    "c1_kernel": (10, 3, 3),
    "c1_stride": (3, 1, 1),
    "c1_bn3d": 16,
    #conv2_1
    "c21_core_counts": 16,
    "c21_kernel": (1, 1, 1),
    "c21_padding": (0, 0, 0),
    "c21_bn3d": 16,
    #conv2_2
    "c22_core_counts": 16,
    "c22_kernel": (3, 1, 1),
    "c22_padding": (1, 0, 0),
    "c22_bn3d": 16,
    #conv2_3
    "c23_core_counts": 16,
    "c23_kernel": (5, 1, 1),
    "c23_padding": (2, 0, 0),
    "c23_bn3d": 16,
    #conv2_4
    "c24_core_counts": 16,
    "c24_kernel": (11, 1, 1),
    "c24_padding": (5, 0, 0),
    "c24_bn3d": 16,
    #conv3_1
    "c31_core_counts": 16,
    "c31_kernel": (1, 1, 1),
    "c31_padding": (0, 0, 0),
    "c31_bn3d": 16,
    #conv3_2
    "c32_core_counts": 16,
    "c32_kernel": (3, 1, 1),
    "c32_padding": (1, 0, 0),
    "c32_bn3d": 16,
    #conv3_3
    "c33_core_counts": 16,
    "c33_kernel": (5, 1, 1),
    "c33_padding": (2, 0, 0),
    "c33_bn3d": 16,
    #conv3_4
    "c34_core_counts": 16,
    "c34_kernel": (11, 1, 1),
    "c34_padding": (5, 0, 0),
    "c34_bn3d": 16,
    #conv4
    "c4_core_counts": 16,
    "c4_kernel": (1, 2, 2),
    "c4_bn3d": 16,
}

for file in all_files_binary_masks:
    name = file.split("\\")[1].split(".mat")[0]
    mask = extract_array_from_matfile(file, "img")
    print(f"mask shape: {mask.shape}")

    hyperparams["n_classes"] = len(np.unique(mask))

    train_model(img = img,
                mask= mask,
                sample_percentage = 0.25,
                hyperparams = hyperparams,
                name = name,
                cnn_params = cnn_params
    )