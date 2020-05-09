import numpy as np
import random
import os
import pickle
import cv2 as cv
import torch
import pickle
from skimage import io, transform
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

def preprocess(dataRoot):
    dataSamples = []
    dataTrainSamples = []
    dataTestSamples = []
    dataValSamples = []

    if not os.path.isdir(os.path.join(dataRoot, 'datasetColor')):
        createDataSet(dataRoot)
    originalDataroot = dataRoot
    dataRoot = os.path.join(dataRoot, 'datasetColor')
    directories = os.listdir(dataRoot)
    directories.sort()
    dataSamples = []
    dataTrainSamples = []
    dataTestSamples = []
    dataValSamples = []
    for i in range(0,len(directories)):
        dir = os.path.join(dataRoot, directories[i])
        files = os.listdir(dir)
        for file in files:
            dataSamples.append({'image': os.path.join(dir,file), 'label' : i})
        dataTrainSamples.extend(dataSamples[:int(0.8*250)])
        dataValSamples.extend(dataSamples[int(0.8*250):300])
        dataTestSamples.extend(dataSamples[300:])
        dataSamples = []

    print('saving train and test data')
    # writing database to file
    with open(os.path.join(originalDataroot, 'train_dataColor.data'), 'wb') as f:
        pickle.dump(dataTrainSamples, f)

    with open(os.path.join(originalDataroot, 'test_dataColor.data'), 'wb') as f:
        pickle.dump(dataTestSamples, f)

    with open(os.path.join(originalDataroot, 'val_dataColor.data'), 'wb') as f:
        pickle.dump(dataValSamples, f)





def createDataSet(dataroot):
    datasetPath = os.path.join(dataroot, 'datasetColor')
    os.mkdir(datasetPath)
    dataSamples = []
    print('iterating dataRoot')
    for root, dirs, files in os.walk(dataroot, topdown=True):
        for name in dirs:
            path = os.path.join(dataroot, name)
            for rootf, dirf, filef in os.walk(path, topdown=True):
                for filen in filef:
                    if not name == "datasetColor" and not name == "dataset":
                        image = Image.open(os.path.join(path,filen))
                        directory = name
                        dataSamples.append(image)
            imgs = doDataAugment(dataSamples)
            path = os.path.join(datasetPath, name)
            if not name == 'datasetColor' and not name == "dataset":
                os.mkdir(path)
            for i , img in enumerate(imgs):
                #img = img.convert("L")   #converts to grayscale
                img.save(os.path.join(path, F'image{i}.jpg'))
            dataSamples = []



def doDataAugment(datasamples):
    finalSamples = []
    loopSamples = []

    for sample in datasamples:
        finalSamples.append(sample)

    loopSamples.extend(datasamples)
    for sample in datasamples:
        transform = T.Compose([T.RandomHorizontalFlip(p=1)])
        img = transform(sample)
        loopSamples.append(img)
        finalSamples.append(img)

    for sample in loopSamples:
        h, w = sample.size
        crop = min(h,w) * 0.9
        transform = T.Compose([T.RandomCrop(size=crop)])
        img = transform(sample)
        finalSamples.append(img)

    # randomcrop all original + horizontalflipped final -> 240 imgs
    for sample in loopSamples:
        h, w = sample.size
        crop = min(h, w) * 0.8
        transform = T.Compose([T.RandomCrop(size=crop)])
        img = transform(sample)
        finalSamples.append(img)


     # randomcrop all original + horizontalflipped final -> 300 imgs
    for sample in loopSamples:
        h, w = sample.size
        crop = min(h, w) * 0.7
        transform = T.Compose([T.RandomCrop(size=crop)])
        img = transform(sample)
        finalSamples.append(img)

    return finalSamples