# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data
import numpy as np
import math
import visdom

import models
import tools

###Experiment parameters###
#You probably want to use https://github.com/IDSIA/sacred now
epochs = 50 #set 0 to run in inference mode only
batch_size = 24 #Lower this if you blowup GPU memory
learning_rate = 0.001

source = "./Makkah/trainMultiComplete.txt"
training_images = range(1,7)
test_images = [0,]

patches = 24 #How many patches to randomly sample from source images
patch_height = 256
patch_width = 256

starting_weights = None #filename or set None for random weight initialization
saved_weights_prefix = 'exp3'

criterion = nn.MSELoss
optimizer = torch.optim.Adam
net = models.skip.Net()

###Initialization###
net.set_upsample((patch_height, patch_width))
viz = visdom.Visdom()
err_plt = None
best_loss = None
if starting_weights:
    net.load_state_dict(torch.load(starting_weights))
optimizer = optimizer(net.parameters(), lr = learning_rate)

if torch.cuda.is_available():
    print("cuda ok")
    net.cuda()
    criterion = criterion().cuda()
else:
    print("no cuda")
    criterion = criterion()

###Data loading###
data = tools.parse_db(source)
tools.autocrop(data)

train_data = tools.tensor_loader([data[i] for i in training_images], patches, patch_height, patch_width, net)
train_size = len(train_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)

test_data = tools.tensor_loader([data[i] for i in test_images], patches, patch_height, patch_width, net)
test_loader = torch.utils.data.DataLoader(test_data, batch_size)

###Training###
for epoch in range(epochs):
    for i, (images, truths) in enumerate(train_loader):
        images = Variable(images.float())
        truths = Variable(truths)
        
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, truths)
        loss.backward()
        optimizer.step()
        
        print('Epoch [%d/%d] Iter [%d/%d] Loss: %.4f'
                  %(epoch+1, epochs, i+1,
                  train_size//batch_size, loss.data[0]))
        #Visdom graphing
        if err_plt:
            viz.updateTrace(X=np.array([(epoch)*train_size//batch_size+i]), Y=np.array([math.log(loss.data[0])]), win = err_plt, append = True, name='0')
        else: #Plot initialization
            err_plt = viz.line(X=np.ones(2),
                               Y=np.array([math.log(loss.data[0]),
                                           math.log(tools.test_loss(test_loader, criterion, net))]))
    ##Model parameter saving##
    t_loss = tools.test_loss(test_loader, criterion, net)
    try:
        if t_loss < best_loss:
            best_loss = t_loss
            best_epoch = epoch+1
            torch.save(net.state_dict(), saved_weights_prefix + '-best_epoch.pkl')
    except TypeError: #Loss value initialization
        best_loss = t_loss
        best_epoch = epoch+1
    torch.save(net.state_dict(), saved_weights_prefix + '-current_epoch.pkl')
    print('Best loss: %.4f (epoch %d)\nCurrent loss: %.4f' % (best_loss, best_epoch, t_loss))
    viz.updateTrace(X=np.array([(epoch+1)*train_size//batch_size]), Y=np.array([math.log(t_loss)]), win = err_plt, append = True, name='1')
net.eval() #Switch to inference mode, only relevant if you use BatchNorm layers

###Inference###
from skimage import io
import matplotlib.pyplot as plt

net.set_upsample(data[0]['image'].shape)
out = net(Variable(torch.from_numpy(data[0]['image'][np.newaxis,np.newaxis,:,:]).cuda().float(), volatile=True)).data.cpu().numpy()[0,0]
cmap = plt.get_cmap('viridis')
io.imsave("test21.png", np.delete(cmap(out), 3, 2))
