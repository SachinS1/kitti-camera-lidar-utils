import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data as data
import numpy as np
from prepare_dataset import KittiData

batch_size = 2

dataset = KittiData()
print(dataset.file_list)
data_loader = data.DataLoader(dataset, batch_size, shuffle=True)
