import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F

from config import *

def get_model(kwargs:dict, cnn_params:dict) -> tuple:
    device = kwargs.setdefault("device", torch.device("cpu"))
    n_classes = kwargs["n_classes"]
    n_bands = kwargs["n_bands"]
    patch_size = kwargs["patch_size"]
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs["ignored_labels"])] = 0.0
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)

    lr = kwargs["learning_rate"]
    center_pixel = True
    model = shortHe(n_bands, n_classes, patch_size, cnn_params)
    model = model.to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = loss_func(weight=kwargs["weights"])

    model = model.to(device)
    epoch = kwargs.setdefault("epoch", 100)
    kwargs.setdefault(
        "scheduler",
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=epoch // 4, verbose=True
        ),
    )
    kwargs.setdefault('scheduler', None)
    kwargs.setdefault("supervision", "full")
    kwargs.setdefault("flip_augmentation", False)
    kwargs.setdefault("radiation_augmentation", False)
    kwargs.setdefault("mixture_augmentation", False)
    kwargs["center_pixel"] = center_pixel
    return model, optimizer, criterion, kwargs

    ##########################################
    ##########################################

class shortHe(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, pathc_size, params:dict):
        super(shortHe, self).__init__()
        self.input_channels = input_channels
        self.patch_size = pathc_size

        self.conv1 = nn.Conv3d(1, params["c1_core_counts"], params["c1_kernel"], params["c1_stride"])
        self.bn_conv1 = nn.BatchNorm3d(params["c1_bn3d"])

        self.conv2_1 = nn.Conv3d(16, params["c21_core_counts"], params["c21_kernel"], padding=params["c21_padding"])
        self.bn_conv2_1 = nn.BatchNorm3d(params["c21_bn3d"])

        self.conv2_2 = nn.Conv3d(16, params["c22_core_counts"], params["c22_kernel"], padding=params["c22_padding"])
        self.bn_conv2_2 = nn.BatchNorm3d(params["c22_bn3d"])

        self.conv2_3 = nn.Conv3d(16, params["c23_core_counts"], params["c23_kernel"], padding=params["c23_padding"])
        self.bn_conv2_3 = nn.BatchNorm3d(params["c23_bn3d"])

        self.conv2_4 = nn.Conv3d(16, params["c24_core_counts"], params["c24_kernel"], padding=params["c24_padding"])
        self.bn_conv2_4 = nn.BatchNorm3d(params["c24_bn3d"])

        self.conv3_1 = nn.Conv3d(16, params["c31_core_counts"], params["c31_kernel"], padding=params["c31_padding"])
        self.bn_conv3_1 = nn.BatchNorm3d(params["c31_bn3d"])

        self.conv3_2 = nn.Conv3d(16, params["c32_core_counts"], params["c32_kernel"], padding=params["c32_padding"])
        self.bn_conv3_2 = nn.BatchNorm3d(params["c32_bn3d"])

        self.conv3_3 = nn.Conv3d(16, params["c33_core_counts"], params["c33_kernel"], padding=params["c33_padding"])
        self.bn_conv3_3 = nn.BatchNorm3d(params["c33_bn3d"])

        self.conv3_4 = nn.Conv3d(16, params["c34_core_counts"], params["c34_kernel"], padding=params["c34_padding"])
        self.bn_conv3_4 = nn.BatchNorm3d(params["c34_bn3d"])

        self.conv4 = nn.Conv3d(16, params["c4_core_counts"], params["c4_kernel"])
        self.bn_conv4 = nn.BatchNorm3d(params["c4_bn3d"])

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            x = self.bn_conv1(x)

            x2_1 = self.conv2_1(x)
            x2_1 = self.bn_conv2_1(x2_1)

            x2_2 = self.conv2_2(x)
            x2_2 = self.bn_conv2_2(x2_2)

            x2_3 = self.conv2_3(x)
            x2_3 = self.bn_conv2_3(x2_3)

            x2_4 = self.conv2_4(x)
            x2_4 = self.bn_conv2_4(x2_4)

            x = x2_1 + x2_2 + x2_3 + x2_4

            x3_1 = self.conv3_1(x)
            x3_1 = self.bn_conv3_1(x3_1)

            x3_2 = self.conv3_2(x)
            x3_2 = self.bn_conv3_2(x3_2)

            x3_3 = self.conv3_3(x)
            x3_3 = self.bn_conv3_3(x3_3)

            x3_4 = self.conv3_4(x)
            x3_4 = self.bn_conv3_4(x3_4)

            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            x = self.bn_conv4(x)

            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.bn_conv1(self.conv1(x)))


        x2_1 = self.conv2_1(x)
        x2_1 = self.bn_conv2_1(x2_1)

        x2_2 = self.conv2_2(x)
        x2_2 = self.bn_conv2_2(x2_2)

        x2_3 = self.conv2_3(x)
        x2_3 = self.bn_conv2_3(x2_3)

        x2_4 = self.conv2_4(x)
        x2_4 = self.bn_conv2_4(x2_4)

        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)

        x3_1 = self.conv3_1(x)
        x3_1 = self.bn_conv3_1(x3_1)

        x3_2 = self.conv3_2(x)
        x3_2 = self.bn_conv3_2(x3_2)

        x3_3 = self.conv3_3(x)
        x3_3 = self.bn_conv3_3(x3_3)

        x3_4 = self.conv3_4(x)
        x3_4 = self.bn_conv3_4(x3_4)

        x = x3_1 + x3_2 + x3_3 + x3_4

        x = F.relu(x)
        x = F.relu(self.bn_conv4(self.conv4(x)))

        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x


def val(net, data_loader, device="cpu", supervision="full"):
    # TODO : fix me using metrics()
    accuracy, total = 0.0, 0.0
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            if supervision == "full":
                output = net(data)
            elif supervision == "semi":
                outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / (total + 0.0001)


def train(
    net,
    optimizer,
    criterion,
    data_loader,
    epochs,
    device,
    scheduler,
    supervision,
    val_loader,
    name,
    display_iter=100,
    display=None,
):
    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)

    save_epoch = epochs // 20 if epochs > 20 else 1

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []

    for e in tqdm(range(1, epochs + 1)):
        # Set the network to training mode
        net.train()
        avg_loss = 0.0

        # Run the training loop for one epoch
        for batch_idx, (data, target) in (enumerate(data_loader)):
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            if supervision == "full":
                output = net(data)
                loss = criterion(output, target)
            elif supervision == "semi":
                outs = net(data)
                output, rec = outs
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[
                    1
                ](rec, data)
            else:
                raise ValueError(
                    'supervision mode "{}" is unknown.'.format(supervision)
                )
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100) : iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
                string = string.format(
                    e,
                    epochs,
                    batch_idx * len(data),
                    len(data) * len(data_loader),
                    100.0 * batch_idx / len(data_loader),
                    mean_losses[iter_],
                )
                update = None if loss_win is None else "append"

                tqdm.write(string)

            iter_ += 1
            del (data, target, loss, output)

        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        if e % save_epoch == 0:
            save_model(
                net,
                name,
            )

def save_model(model, name):
    model_path = f"./checkpoints/models/{name}.pth"
    tqdm.write(f"Saving neural network weights in {model_path}")
    torch.save(model.state_dict(), model_path)

