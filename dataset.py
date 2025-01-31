import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.io import loadmat
import os
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt as bwdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class COCO(Dataset):
    def __init__(self, rootDir, offline=False):
        self.rootDirImg = rootDir + "images/"
        self.rootDirGt = rootDir + "groundTruth/" + "person/" + "skeletons/"
        self.rootDirGtEdges = rootDir + "groundTruth/" + "person/" + "edges/"
        self.listData = sorted(os.listdir(self.rootDirGt))
    def __len__(self):
        return len(self.listData)
                
    def __getitem__(self, i):
        # input and target images
        inputName = self.listData[i].replace('.png','.jpg')
        targetName = self.listData[i]
        # process the images
        transf = transforms.ToTensor()
        inputImage = transf(Image.open(self.rootDirImg + inputName).convert('RGB'))

        tensorBlue = (inputImage[0:1, :, :] * 255.0) - 104.00698793
        tensorGreen = (inputImage[1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (inputImage[2:3, :, :] * 255.0) - 122.67891434
        inputImage = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 0)

        targetImage = transf(Image.open(self.rootDirGt + targetName).convert('L'))*255.0
        #edge = transf(Image.open(self.rootDirGtEdges + targetName).convert('L')).squeeze_(0).numpy()> 0.5
        #dist = 2.0*bwdist(1.0 - (edge.astype(float)))
        #make_scale = np.vectorize(lambda x, y: 0 if y < 0.99 else x)

        #scale = make_scale(dist,targetImage)
        #targetImage = torch.from_numpy(scale).float().unsqueeze_(0)
        return inputImage, targetImage

class SKLARGE(Dataset):
    def __init__(self, rootDir, pathList):
        self.rootDir = rootDir
        self.listData =  pd.read_csv(pathList, dtype=str, delimiter=',')
        self.new_size = []

    def __len__(self):
        return len(self.listData)
                
    def __getitem__(self, i):
        # input and target images
        inputName = self.listData.iloc[i, 0]
        targetName = self.listData.iloc[i, 1]
        # process the images
        transf = transforms.ToTensor()
        inputImage = Image.open(inputName).convert('RGB')

        if len(self.new_size) == 0:
          self.new_size = inputImage.size
        
        inputImage = inputImage.resize(self.new_size)
        inputImage = transf(inputImage)
        tensorBlue = (inputImage[0:1, :, :] * 255.0) - 104.00698793
        tensorGreen = (inputImage[1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (inputImage[2:3, :, :] * 255.0) - 122.67891434
        inputImage = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 0)

        """ CODE FOR RAW .MAT FILES
        itemGround = loadmat(self.rootDir + targetName)
        edge, skeleton = itemGround['edge'], itemGround['symmetry']
        dist = bwdist(1.0 - edge.astype(float))
        make_scale = np.vectorize(lambda x, y: 0 if y < 0.5 else x)
        #These should be parameters of the class
        receptive_fields = np.array([14,40,92,196])
        p = 1.2
        ####
        quantization = np.vectorize(lambda s: 0 if s < 0.001 else np.argmax(receptive_fields > p*s) + 1)
        scale = make_scale(dist,skeleton)
        quantise = quantization(scale)
        scaleTarget = torch.from_numpy(scale).float()
        quantiseTarget = torch.from_numpy(quantise)
        """
        targetImage = transf(Image.open(targetName).convert('L').resize(self.new_size))*255.0
        return inputImage, targetImage

class SKLARGE_RAW(Dataset):
    def __init__(self, rootDirImg, rootDirGt):
        self.rootDirImg = rootDirImg
        self.rootDirGt = rootDirGt
        self.listData = [sorted(os.listdir(rootDirImg)),sorted(os.listdir(rootDirGt))]

    def __len__(self):
        return len(self.listData[1])
                
    def __getitem__(self, i):
        # input and target images
        inputName = self.listData[0][i]
        targetName = self.listData[1][i]
        # process the images
        transf = transforms.ToTensor()
        inputImage = transf(Image.open(self.rootDir + inputName).convert('RGB'))
        itemGround = loadmat(self.rootDir + targetName)
        edge, skeleton = itemGround['edge'], itemGround['symmetry']
        dist = bwdist(1.0 - edge.astype(float))
        make_scale = np.vectorize(lambda x, y: 0 if y < 0.5 else x)
        scale = make_scale(dist,skeleton)
        scaleTarget = torch.from_numpy(scale).float()

        return inputImage, scaleTarget


class SKLARGE_TEST(Dataset):
    def __init__(self, rootDirImg):
        self.rootDirImg = rootDirImg
        self.listData = sorted(os.listdir(rootDirImg))

    def __len__(self):
        return len(self.listData)
                
    def __getitem__(self, i):
        # input and target images
        inputName = self.listData[i]
        # process the images
        transf = transforms.ToTensor()
        inputImage = transf(Image.open(self.rootDirImg + inputName).convert('RGB'))
        inputName = inputName.split(".jpg")[0] + ".png"
        return inputImage, inputName
