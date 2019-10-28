# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:26:30 2019

@author: 11104510
"""

import torch
import torch.utils.data as Data
torch.manual_seed(1)#使得每次运行获得的随机数相同

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        )

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        #train your data...
        print('Epoch: ', epoch, '| step: ', step, '| batch x: ', batch_x.numpy(),
              '| batch y: ', batch_y.numpy())
        
