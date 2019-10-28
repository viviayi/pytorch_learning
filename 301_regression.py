# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 19:02:22 2019

@author: 11104510
"""

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)#转换为列向量
y = x.pow(2) + 0.2*torch.rand(x.size())

#plt.scatter(x.data.numpy(), y.data.numpy())
#plt.show()

class Net(torch.nn.Module):#继承torch的Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()#继承__init__
        #定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)#隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)#输出层线性输出
        
    def forward(self, x):
        #正向传播输入值，神经网络分析输出值
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
    
net = Net(n_feature=1, n_hidden=10, n_output=1)

#print(net)

#optimizer是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)#传入net的所有参数，学习率
loss_func = torch.nn.MSELoss()#预测值和真实值的误差计算公式（均方差）

plt.ion()
plt.show()

for t in range(200):
    prediction = net(x)#喂给net训练数据x,输出预测值
    
    loss = loss_func(prediction, y)#计算两者误差
    
    optimizer.zero_grad()#清空上一步的残余更新参数值
    loss.backward()#误差反向传播，计算参数更新值
    optimizer.step()#将参数更新值施加到net的parameters上
    
    if t%5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
