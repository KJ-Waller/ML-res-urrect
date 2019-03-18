#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# In[2]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[40]:


# Image preprocessing modules
transform = torchvision.transforms.Compose([
    torchvision.transforms.Pad(4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(32),
    torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10('./data/', train=True, transform=transform, download=True)
test_set = torchvision.datasets.CIFAR10('./data/', train=False, transform=transform)


# In[41]:


num_epochs = 5
batch_size = 50
num_classes = 10
learning_rate = 0.001 #68%
# learning_rate = 0.0001 #61%


# In[50]:


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.inchannels = in_channels
        self.outchannels = out_channels
        
    def forward(self, x):
        residual = x
        
        if self.inchannels != self.outchannels:
            residual = self.conv(x)
            
        x = self.layer1(x)
        x += residual
        x = F.relu(x)
        
        return x
        
    
    
class CIFARNET(nn.Module):
    
    def __init__(self, num_classes=10):
        super(CIFARNET, self).__init__()
        
        self.reslayer1 = nn.Sequential(
            ResBlock(3, 128),
            ResBlock(128, 128),
            ResBlock(128, 256),
            nn.MaxPool2d(2),
            ResBlock(256, 128),
            nn.MaxPool2d(2),
            ResBlock(128, 128),
            nn.MaxPool2d(2)
        )
        
#         self.layer1 = nn.Sequential( #74.08 %
#             nn.Conv2d(3, 128, 3, 1, 1),
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128, 256, 3, 1, 1),
#             nn.Conv2d(256, 256, 3, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.MaxPool2d(2), # Output 4
#             nn.Conv2d(256, 128, 3, 1, 1),
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(2), # Output 4
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(2), # Output 2
#         )
        
        self.fc1 = nn.Linear(4*4*128, 4096)
        self.fc2 = nn.Linear(4096, 1000)
        self.fc3 = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.reslayer1(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
#         x = F.relu(self.fc3(x))
        return x
    
model = CIFARNET(num_classes).to(device)


# In[51]:


train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


# In[52]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[53]:


total_step = len(train_loader)

model.train()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        output = model(images)
        loss = criterion(output, labels)
        
        # Backprop and Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print out current progress of training
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# In[54]:


# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


# In[ ]:




