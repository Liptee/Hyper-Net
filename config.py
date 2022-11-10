import torch.nn as nn

path_to_img = ""
index_in_img_file = ""

path_to_mask = "dataset/first_row_mask.mat"
index_in_mask_file = "img"

path_to_img = "dataset/first_row_img_no_blue.mat"
index_in_img_file = "image"

cuda_device = "cpu"

loss_func = nn.CrossEntropyLoss
#loss_func = nn.MultiLabelSoftMarginLoss


weight_decay = 0.01
###hyperparams:
epochs = 1
patch_size = 7
batch_size = 40
learning_rate = 0.01

# conv1
c1_core_counts = 16
c1_kernel = (10, 3, 3) #deep height width
c1_stride = (3, 1, 1)
c1_bn3d = 16

# conv2_1
c21_core_counts = 16
c21_kernel = (1, 1, 1) #deep height width
c21_padding = (0, 0, 0)
c21_bn3d = 16

# conv2_2
c22_core_counts = 16
c22_kernel = (3, 1, 1) #deep height width
c22_padding = (1, 0, 0)
c22_bn3d = 16

# conv2_3
c23_core_counts = 16
c23_kernel = (5, 1, 1) #deep height width
c23_padding = (2, 0, 0)
c23_bn3d = 16

####train_gt mode:
train_gt_mode = "disjoint" #default
# train_gt_mode = "random"
# train_gt_mode = "fixed"