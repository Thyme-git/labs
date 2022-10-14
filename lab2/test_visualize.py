from pyexpat import model
from unittest import TestLoader
import torch
import torch.nn as nn

from torch.utils import data
from tqdm import tqdm

from tvid import TvidDataset
from detector import Detector

from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

tl = data.DataLoader(TvidDataset(root='./tiny_vid', mode='test', transform=False),
                                 batch_size=16, shuffle=True, num_workers=4)

tf = transforms.Compose([
     transforms.ToPILImage(),
     transforms.Resize([224,224]),
     transforms.ToTensor(),
     transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

model = torch.load('./model100')

fig, ax = plt.subplots(nrows=4, ncols=4, sharex='all', sharey='all')
fig.suptitle('Plot predicted samples')
ax = ax.flatten()

class_list = ['bird', 'car', 'dog', 'lizard', 'turtle']

with torch.no_grad():
    X, y = next(iter(tl))
    X_in = tf(X[0])
    X_in = torch.unsqueeze(X_in, dim=0)
    for i in range(1, 16):
        X_in = torch.cat([X_in, torch.unsqueeze(tf(X[i]), dim=0)], dim=0)
    
    X_in = X_in.to('cuda')
    cls_pred, box_pred = model(X_in)
    box = y[:, 1:]
    cls = y[:, 0].long()
    cls_pred = cls_pred.argmax(1)

    box_pred = box_pred.to('cpu')
    cls_pred = cls_pred.to('cpu')

    for i in range(16):
        rect = plt.Rectangle((box[i][0], box[i][1]), box[i][2]-box[i][0], box[i][3]-box[i][1], fill=False, color='red')
        rect_pred = plt.Rectangle((box_pred[i][0], box_pred[i][1]), box_pred[i][2]-box_pred[i][0], box_pred[i][3]-box_pred[i][1], fill=False, color='blue')
        ax[i].set_title(class_list[cls_pred[i]])
        ax[i].imshow(transforms.ToPILImage()(X[i]))
        ax[i].add_patch(rect)
        ax[i].add_patch(rect_pred)

plt.show()
        
                                 