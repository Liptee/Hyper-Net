import os
import numpy as np
from glob import glob
from scipy.io import loadmat, savemat
from concurrent.futures import ThreadPoolExecutor

def load_data(path:str, ending:str) -> list:
    X = sorted(glob(os.path.join(path, f"*{ending}")))
    return X

def extract_array_from_matfile(file:str,index:str) -> np.ndarray:
    matfile = loadmat(file)
    return matfile[index]

def thread_binaring(label:int, mask:np.ndarray, file:str, name:str, idx:str):
    copy = np.copy(mask)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if copy[i][j] == label:
                copy[i][j] = 2
            else: copy[i][j] = 1

    copy[0][0] = 0
    orig = loadmat(file)
    orig[idx] = copy
    savemat(f"checkpoints/bi_masks/{name}_{label}.mat", orig)

def create_binary_masks(file:str, idx:str, name="binary"):
    mask = extract_array_from_matfile(file, idx)
    num_labels = len(np.unique(mask))

    with ThreadPoolExecutor(max_workers=num_labels) as executor:
        for label in range(1, num_labels):
            executor.submit(thread_binaring, label, mask, file, name, idx)