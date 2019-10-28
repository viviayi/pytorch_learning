# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:11:19 2019

@author: 11104510
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)#转换为列向量
y = x.pow(2) + 0.2*torch.rand(x.size())

plt.figure(1, figsize=(10,3))

def save():
    net1 = torch.nn.Sequential(
            torch.nn.Linear(1,10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,1)
            )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()
    
    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    
    torch.save(net1, 'net.pkl')
    torch.save(net1.state_dict(), 'net_params.pkl')
    
    
def restore_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)
    
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    
def restore_params():
    net3 = torch.nn.Sequential(
            torch.nn.Linear(1,10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,1)
            )
    
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)
    
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    
save()
restore_net()
restore_params()

plt.show()