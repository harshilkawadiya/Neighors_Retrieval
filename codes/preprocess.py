import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from scipy.misc import imresize
import csv
import torch
import skimage.io as io
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms as trn
from PIL import Image
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
preprocess = trn.Compose([
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

import nltk
from nltk.corpus import wordnet as wn

K=10 #selection of the retrieval number


traindir = '/scratch/harshil.j/Retrieval/Flicker8k/Flickr8k_Dataset/train'
valdir = '/scratch/harshil.j/Retrieval/Flicker8k/Flickr8k_Dataset/test'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        print img.size()
        return img,target,path.split('/')[-1]



train_dataset = MyImageFolder(
    traindir,
    transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

val_dataset = MyImageFolder(
    valdir,
    transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ]))


vgg16 = models.vgg16('pretrained=true')
vgg16=vgg16.cuda()
print 'Model Parameters Loaded'

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1)


class phrase:
    def __init__(self):
        self.type = 0
        self.value = ''
# tr_data used as retrieval data.
# te_data used as query data for joint probability computation.

tr_data=[]
te_data=[]
tr_images=[]
te_images=[]
# base='/scratch/harshil.j/Retrieval/Flicker8k/Flickr8k_Dataset/Flicker8k_Dataset/'
# f=open('/scratch/harshil.j/Retrieval/Flicker8k/Flickr8k_Dataset/Flickr_8k.trainImages.txt','r+')
# import cv2
for i, (input,target,path) in enumerate(train_loader):
	print i
	temp=[]
	tr_images.append(path)
	feat = vgg16(Variable(input).cuda())
	feats = feat.cpu().data.numpy()
	tr_data.append(feats)

tr_data = np.array(tr_data)
tr_data = np.reshape(tr_data,(len(tr_data),-1))

for i, (input,target,path) in enumerate(val_loader):
	temp=[]
	te_images.append(path)
	feat = vgg16(Variable(input).cuda())
	feats = feat.cpu().data.numpy()
	te_data.append(feats)

te_data = np.array(te_data)
te_data = np.reshape(te_data,(len(te_data),-1))


# feature normalization
# tr_data,tr_images,te_data,te_images
import sklearn.preprocessing
from sklearn.preprocessing import normalize as nm
tr_data1=nm(tr_data,norm='l2')
te_data1=nm(te_data,norm='l2')


# # queries are treated as testing set and training data set are treated as retrieval set.
# # testing set images - 1000 and training set/retrieval set - 5000
# # K - 10
# # get k neighors,distances and indices from te_data of re_set.
from sklearn.neighbors import KDTree
kdt = KDTree(tr_data1, leaf_size=30, metric='euclidean')
distances,indices=kdt.query(tr_data1, k=6, return_distance=True) 


final_tr_data = []
for i in range(0,len(indices)):
	merge = []
	for j in range(0,6):
		merge.append(tr_data[indices[i][j]])
	merge = np.array(merge)
	merge = np.reshape(merge,(6000))
	final_tr_data.append(merge)

import csv
myFile = open('/scratch/harshil.j/train.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(final_tr_data)

myFile.close()

kdt = KDTree(te_data1, leaf_size=30, metric='euclidean')
distances,indices=kdt.query(te_data1, k=6, return_distance=True) 


final_te_data = []
for i in range(0,len(indices)):
	merge = []
	for j in range(0,6):
		merge.append(te_data[indices[i][j]])
	merge = np.array(merge)
	merge = np.reshape(merge,(6000))
	final_te_data.append(merge)

import csv
myFile = open('/scratch/harshil.j/test.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(final_te_data)

myFile.close()

# # second term computation in joint probability P(y,I)
# def prb_I_given_J(dists):
# 	sigma = np.sum(dists)
# 	sigma=sigma+0.00000000000001
# 	dists=dists/sigma
# 	return dists

# # Data Management
imgpath = open('/home/harshil.j/codes/Harshil/image_names.txt','r+')
img_id=[]
inverse_map={}
count=0
for lines in imgpath:
	image_n=lines[0:-1]
	img_id.append(image_n)
	inverse_map[image_n]=count
	count=count+1


data_phrases={}

# File Parsing for the extracted file from Stanford Parser. 

dct={}
cnt=0
prev = 0
orde={}
for j in range(17):
	f_name="/home/harshil.j/codes/Triple Extraction/Outputs/processed_data"+str(j)+".txt.xml"
	f=open(f_name,'r+')
	for line in f:
		category=0
		if line.startswith('#####'):
			cnt=cnt+1
			category=-1
		if(line.startswith("attribute-object")):
			category=1
		elif(line.startswith("object-verb-object")):
			category=2
		elif(line.startswith("object-verb")):
			category=3
		if category>0:
			value=line.split('\t')[1]
			if value not in orde:
				orde[value] = 1
		if cnt%5 ==0 and cnt!=0 and category==-1:
			for k in orde.keys():
				if k in dct:
					dct[k]+=1
				else:
					dct[k]=1
			orde={}

dec={}
it=1
import operator
sorted_x = sorted(dct.items(), key=operator.itemgetter(1))
phrases = []
for temp in sorted_x:
	if temp[1]>=10:
		phrases.append(temp[0])
		dec[temp[0]]=it
		it=it+1

myFile = open('/scratch/harshil.j/phrases.txt', 'w')
for temp in phrases:
	myFile.write(temp)


myFile.close()

lst = [0]*len(dec)
data_phrases=[]
for j in range(17):
	f_name="/home/harshil.j/codes/Triple Extraction/Outputs/processed_data"+str(j)+".txt.xml"
	f=open(f_name,'r+')
	for line in f:
		category=0
		if line.startswith('#####'):
			cnt=cnt+1
			category=-1
		ct=cnt/5 + 1
		if(line.startswith("attribute-object")):
			category=1
		elif(line.startswith("object-verb-object")):
			category=2
		elif(line.startswith("object-verb")):
			category=3
		if category>0:
			value=line.split('\t')[1]
			if value not in orde:
				orde[value] = 1
		if cnt%5 ==0 and cnt!=0 and category==-1:
			print 'Here'
			for k in orde.keys():
				if k in dec:
					lst[dec[k]-1]=1
			data_phrases.append(lst)
			lst = [0]*len(dec)
			orde={}

train_phrases=[]
for img_n in tr_images:
	train_phrases.append(data_phrases[inverse_map[img_n[0]]])

test_phrases=[]
for img_n in te_images:
	test_phrases.append(data_phrases[inverse_map[img_n[0]]])

myFile = open('/scratch/harshil.j/train_ph.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(train_phrases)

myFile.close()



myFile = open('/scratch/harshil.j/test_ph.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(test_phrases)

myFile.close()


# print len(dec)


# # Third term computation in the formula
# def prob_y_given_J(data_te,data_tr):
# 	tr_scr=[]
# 	for j in range (0,len(data_tr)):
# 		curr_scr=0
# 		for i in data_te.keys():
# 			q_phrase=data_te[i]
# 			for k in data_tr[j].keys():
# 				tr_phrase=data_tr[j][k]
# 				for q in q_phrase:
# 					for t in tr_phrase:
# 						q_set1=q.value.split(' ')
# 						q_set2=t.value.split(' ')
# 						score=0
# 						if(q.type==t.type):
# 							for idx1 in range(len(q_set1)):
# 								comb=0
# 								for idx2 in range(len(q_set2)):
# 									# print q_set1[idx1],q_set2[idx2]
# 									syn1=wn.synsets(q_set1[idx1])
# 									syn2=wn.synsets(q_set2[idx2])
# 									if len(syn1)>0 and len(syn2)>0:
# 										comb=max(comb,syn1[0].path_similarity(syn2[0]))
# 								score=score+comb
# 							curr_scr=max(curr_scr,score)
# 		tr_scr.append(curr_scr)
# 	return tr_scr

# # final scores calculation
# retrieval_score={}
# for k in range(2):
# 	if k%10==0:
# 		print k+1,'set image getting processed...'
# 	for i in range(100,200):
# 		if i%100==0:
# 			print i,'th image getting transacted....'
# 		idx=inverse_map[te_images[i]]
# 		tr_phrases=[]
# 		for j in range(K):
# 			idx2=inverse_map[tr_images[indices[i][j]]]
# 			tr_phrases.append(data_phrases[idx2])
# 		pyj_scores=prob_y_given_J(data_phrases[k],tr_phrases)	
# 		norm_distances=prb_I_given_J(distances[i])
# 		joint_prob=0
# 		for j in range(K):
# 			joint_prob+=norm_distances[j]*pyj_scores[j]*(0.1)
# 		if k not in retrieval_score.keys():
# 			retrieval_score[k]={}
# 		retrieval_score[k][i]=joint_prob

# print retrieval_score
# top_k_images={}

