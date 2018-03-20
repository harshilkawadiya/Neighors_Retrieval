import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet as wn

K=5 #selection of near neighors in order to feed to model

class TableModule(nn.Module):
    def __init__(self):
        super(TableModule, self).__init__()        
    def forward(self, x, dim):
        y = torch.cat(x, dim)
        return y


class VGGkModule(nn.Module):
    def __init__(self):
        super(VGGkModule,self).__init__()
        self.pool = nn.MaxPool1d(5)
    def forward(self,x):
        x = x.view(1000,-1,5)
        out = self.pool(x)
        return out.view(-1,1000)
    def backward(self,grad_out):
        gr_in = grad_out.clone()
        return gr_in



class ParallelModule(nn.Module):
    def __init__(self):
        super(ParallelModule, self).__init__()
        self.layer2 = VGGkModule()
    def forward(self, x):
        x  = x.view(6,-1,1000)
        y1 = x[0]
        y2 = self.layer2(x[1:6])
        par = TableModule()
        y  = par([y1,y2],1)
        return y


class ProjModule(nn.Module):
    def __init__(self):
        super(ProjModule,self).__init__()
        self.layer1 = nn.Linear(2000,698)
        self.net = ParallelModule()
    def forward(self,x):
        x1 = self.net(x)
        y = self.layer1(x1)
        out = F.sigmoid(y)
        return out


model = ProjModule().cuda()
criterion = nn.BCELoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), 0.01,
                            momentum=0.9,
                            weight_decay=1e-2)

# Data loading code
traindir = '/scratch/harshil.j/train.csv'
testdir = '/scratch/harshil.j/test.csv'
trainlabel = '/scratch/harshil.j/train_ph.csv'
testlabel = '/scratch/harshil.j/test_ph.csv'


class MyImageFolder(datasets.ImageFolder):
    def __init__(self,traindir,trainlabel):
        self.imgs = pd.read_csv(traindir,header=None).as_matrix().astype('float')
        self.tgt = pd.read_csv(trainlabel,header=None).as_matrix().astype('float')
    def __getitem__(self, index):
        img = self.imgs[index]
        target  = self.tgt[index]
        img= np.reshape(img,(6,1000))
        img= torch.from_numpy(img)
        target= torch.from_numpy(target)
        return img, target

class MyValFolder(datasets.ImageFolder):
    def __init__(self,testdir,testlabel):
        self.imgs = pd.read_csv(testdir,header=None).as_matrix().astype('float')
        self.tgt = pd.read_csv(testlabel,header=None).as_matrix().astype('float')
    def __getitem__(self, index):
        img = self.imgs[index]
        target  = self.tgt[index]
        img= np.reshape(img,(6,1000))
        return img, target

train_dataset = MyImageFolder(
    traindir,trainlabel)

test_dataset = MyValFolder(
    testdir,testlabel)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=10,pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1,pin_memory=True)


err = 0.0
iter = 0

model.train()
for epoch in range(1,50):
    for i, (input,target) in enumerate(train_loader):
        input_var = torch.autograd.Variable(input.float()).cuda()
        target_var = torch.autograd.Variable(target.float()).cuda()
        optimizer.zero_grad()
        output = model(input_var)
        op = (output>0.8).byte().cuda()
        loss = criterion(output, target_var)
        loss.backward()
        target_var = target_var.byte()
        scr = (op != target_var).sum() 
        hamming_score = 1 - float(scr.data[0])/6980
        optimizer.step()
        iter = iter+1
        if iter % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} | {}'.format(epoch, i * len(input),len(train_loader.dataset),100* i /len(train_loader), loss.data[0],hamming_score))
            # iter = 0?


model.eval()
for epoch in range(1,10):
    for i, (input,target) in enumerate(test_loader):
        input_var = torch.autograd.Variable(input.float()).cuda()
        target_var = torch.autograd.Variable(target.float()).cuda()
        optimizer.zero_grad()
        output = model(input_var)
        op = (output>0.8).byte().cuda()
        loss = criterion(output, target_var)
        loss.backward()
        target_var = target_var.byte()
        scr = (op != target_var).sum() 
        hamming_score = 1 - float(scr.data[0])/6980
        optimizer.step()
        iter = iter+1
        if iter % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} | {}'.format(epoch, i * len(input),len(train_loader.dataset),100* i /len(train_loader), loss.data[0],hamming_score))

torch.save(model,'./model.rar')
# To reload the model use:
model1 =torch.load('./model.rar')

results = []
model.eval()
for i, (input,target) in enumerate(test_loader):
    input_var = torch.autograd.Variable(input.float()).cuda()
    target_var = torch.autograd.Variable(target.float()).cuda()
    output = model(input_var)
    out = output.data.cpu().numpy()
    temp = out[0].argsort()[-3:][::-1]
    lst = [0]*698
    for clc in temp:
        lst[clc] = 1
    results.append(lst)

myFile = open('/scratch/harshil.j/results_temp.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(results)

myFile.close()
