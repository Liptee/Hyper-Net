import torch
import numpy as np
import torch.utils.data as data

from scripts.model import train
from scripts.HyperX import HyperX
from scripts.model import get_model
from scripts.utils import sample_gt


from config import *

def train_model(img:np.ndarray,
                mask:np.ndarray,
                sample_percentage:float,
                hyperparams:dict,
                name:str,
                cnn_params:dict,
                weights_path = None):

    model, optimizer, loss, hyperparams = get_model(hyperparams, cnn_params=cnn_params)
    
    if weights_path:
        model.load_state_dict(torch.load(weights_path))

    train_gt, _ = sample_gt(mask, sample_percentage, mode='disjoint')
    train_gt, val_gt = sample_gt(train_gt, 0.95, mode="disjoint")

    train_dataset = HyperX(img, train_gt, **hyperparams)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size = hyperparams["batch_size"],
        shuffle=True,
    )

    val_dataset = HyperX(img, val_gt, **hyperparams)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=hyperparams["batch_size"],
    )

    print(type(optimizer))
    train(
        net=model,
        optimizer=optimizer,
        criterion=loss,
        data_loader=train_loader,
        epochs=hyperparams["epochs"],
        device=hyperparams["device"],
        scheduler=hyperparams["scheduler"],
        supervision=hyperparams["supervision"],
        val_loader=val_loader,
        name = name,
    )