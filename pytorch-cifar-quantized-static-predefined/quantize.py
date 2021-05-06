import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

net = torchvision.models.quantization.resnet18(pretrained=False, quantize=False)
net.load_state_dict(torch.load('resnet18_weights_predefined.pth'))

