# coding: utf-8
# In[1]:

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# ## Hyper Parameters





# ## Build CustomedDataset
# build this class in order to load data more conveniently and to process data in the way of mini-batch

# In[9]:

class CustomedDataSet(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.train = train
        if self.train :
            trainX = pd.read_csv('./data/train.csv')
            trainY = trainX.label.values.tolist()
            trainX = trainX.drop('label',axis=1).values.reshape(trainX.shape[0], 1, 28, 28)
            self.datalist = trainX
            self.labellist = trainY
        else:
            testX = pd.read_csv('./data/test.csv')
            testX = testX.values.reshape(testX.shape[0], 1, 28, 28)
            self.datalist = testX
            
    def __getitem__(self, index):
        if self.train:
            return torch.Tensor(self.datalist[index].astype(float)),self.labellist[index]
        else:
            return torch.Tensor(self.datalist[index].astype(float))
    
    def __len__(self):
        return self.datalist.shape[0]



# ## CNN Model
# 2 conv layers + 1 fc layer  
# with batchnorm and the activation function of ReLU

# In[13]:

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1 ,32, kernel_size=3,padding=1),# 30
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32 ,48, kernel_size=3,padding=1), # 32
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.MaxPool2d(2))
        self.dropout = nn.Dropout2d(0.2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=2,padding=1), # 35
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(1024, 128)
        self.fc1 = nn.Linear(128, 10)

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer3(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)

        return out

def main():
        # In[2]:

    num_epochs = 150
    batch_size = 1600
    learning_rate = 0.0015
        
    # In[10]:

    train_dataset = CustomedDataSet()


    # In[11]:

    test_dataset = CustomedDataSet(train=False)


    # In[12]:

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

    # In[14]:

    cnn = CNN()
    cnn.cuda()


    # In[15]:

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),lr=learning_rate)


    # In[16]:

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            
            #Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            if (i+1) % 50 == 0:
                # iterate the whole dataset with batch size,
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.10f' %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data.item()))


    # ## apply our trained model to test dataset

    # In[17]:

    cnn.eval()  #    Sets the module in evaluation mode.
                #   This has any effect only on modules such as Dropout or BatchNorm.


    # In[18]:

    ans = torch.cuda.LongTensor()    #build a tensor to concatenate answers


    # In[19]:

    #I just can't throw all of test data into the network,since it was so huge that my GPU memory cann't afford it
    for images in test_loader:
        images = Variable(images).cuda()
        outputs = cnn(images)
        _,predicted = torch.max(outputs.data, 1)
        ans = torch.cat((ans,predicted),0)


    # In[20]:

    ans = ans.cpu().numpy()              #only tensor on cpu can transform to the numpy array


    # In[21]:

    aa = pd.DataFrame(ans)
    aa.columns = ['Label']
    Id = range(1,aa.size+1)
    aa.insert(0, 'ImageId', Id)               #bulid the summit csv

    # In[23]:

    aa.to_csv('submit_pytorch.csv',index = False)



if __name__ =='__main__':
    main()