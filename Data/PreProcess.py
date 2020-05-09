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

#OUTPUT_SIZE = (100, 100)



def preprocess(dataRoot, normalise=False):
    # get images and corresponding labels in a dictionary
    dataSamples = []
    dataTrainSamples = []
    dataTestSamples = []
    dataValSamples = []
    labels = []
    samples = 40
    idx = 0
    print('iterating dataRoot')
    for root, dirs, files in os.walk(dataRoot, topdown=True):
        for name in dirs:
            for rootf, dirf, filef in os.walk(os.path.join(dataRoot, name), topdown=True):
                for filen in filef:
                    #if idx < samples:
                    image = io.imread(os.path.join(dataRoot, name, filen), as_gray=True)
                    image = Image.fromarray(image)
                    label = name
                    dataSamples.append({'image': image, 'label': label})
                    #idx += 1
                #idx = 0
                # to later convert labels to one hot vectors
                if not labels.__contains__(name):
                    labels.append(name)
            dataSamples = doDataAugment(dataSamples)
            random.shuffle(dataSamples)
            for sample in dataSamples[:int(0.8*300)]:
                dataTrainSamples.append(sample)
            for sample in dataSamples[int(0.8*300):300]:
                dataValSamples.append(sample)
            for sample in dataSamples[300:]:
                dataTestSamples.append(sample)
            print(len(dataSamples))
            dataSamples = []

    print('converting labels to onehotvectors')
    # converts label names to one hot vectors and saves dictionary
    labels = to_one_hot_vector(labels)
    with open(os.path.join(dataRoot, 'label_dictionary.data'), 'wb') as f:
        pickle.dump(labels, f)

    print('converting sample labels to one hot vectors')
    # converts dataSampleLabels to one hot vectors according to dictionary
    dataTrainSamples = sample_labels_to_one_hot_vector(dataTrainSamples, labels)
    dataTestSamples = sample_labels_to_one_hot_vector(dataTestSamples, labels)
    dataValSamples = sample_labels_to_one_hot_vector(dataValSamples, labels)


    print('applying transforms')
# Transforms
    # rescale
    # randomCrop
    # toTensor
    #for sample in dataTrainSamples:
        #rescale(sample['image'], OUTPUT_SIZE)
        #randomCrop(image, CROPOUTPUTSIZE)
        #ToTensor(sample['image'])

    #for sample in dataTestSamples:
        #rescale(sample['image'], OUTPUT_SIZE)
        # randomCrop(image, CROPOUTPUTSIZE)
        #ToTensor(sample['image'])


    print('shuffling data')
#shuffle data
    random.shuffle(dataTrainSamples)
    random.shuffle(dataTestSamples)
    random.shuffle(dataValSamples)

    print('saving train and test data')
#writing database to file
    with open(os.path.join(dataRoot, 'train_data.data'), 'wb') as f:
        pickle.dump(dataTrainSamples, f)

    with open(os.path.join(dataRoot, 'test_data.data'), 'wb') as f:
        pickle.dump(dataTestSamples, f)

    with open(os.path.join(dataRoot, 'val_data.data'), 'wb') as f:
        pickle.dump(dataValSamples, f)


def doDataAugment(dataSamples):
    finalSamples = []
    loopSamples = []

    for sample in dataSamples:
        #finalSamples.append({'image' : np.asarray(sample['image']), 'label' : sample['label']})
        finalSamples.append({'image': sample['image'], 'label': sample['label']})

    loopSamples.extend(dataSamples)
    #horizontal flip all original images   when 40 imgs:   data -> 80 imgs, final -> 80 imgs
    for sample in dataSamples:
        transform = T.Compose([T.RandomHorizontalFlip(p=1)])
        img = transform(sample['image'])
        loopSamples.append({'image': img, 'label': sample['label']})
        finalSamples.append({'image': (img), 'label': sample['label']})

    #randomcrop all original + horizontalflipped final -> 160 imgs
    for sample in loopSamples:
        h, w = sample['image'].size
        transform = T.Compose([T.RandomCrop(size=(w-20, h - 20))])
        img = transform(sample['image'])
        dict = {'image': (img), 'label': sample['label']}
        finalSamples.append(dict)

    # randomcrop all original + horizontalflipped final -> 240 imgs
    for sample in loopSamples:
        h, w = sample['image'].size
        transform = T.Compose([T.RandomCrop(size=min(w - 10, h - 10))])
        img = transform(sample['image'])
        dict = {'image': img, 'label': sample['label']}
        finalSamples.append(dict)


     # randomcrop all original + horizontalflipped final -> 300 imgs
    for sample in loopSamples:
        h, w = sample['image'].size
        transform = T.Compose([T.RandomCrop(size=(w - 40, h - 40))])
        img = transform(sample['image'])
        dict = {'image': (img), 'label': sample['label']}
        finalSamples.append(dict)

    return finalSamples







#def ToTensor(image):
    #image = torch.from_numpy(image)
    #return image



#def randomCrop(image, cropOutputSize):



# output_size (tuple or int): Desired output size. If tuple, output is
# matched to output_size. If int, smaller of image edges is matched
# to output_size keeping aspect ratio the same.



def to_one_hot_vector(labels):
    #vectors = np.eye(labels.__len__())
    for i in range(labels.__len__()):
        #labels[i] = {"label": labels[i], "vector": vectors[i]}
        labels[i] = {"label": labels[i], "vector": i}
    return labels


def sample_labels_to_one_hot_vector(dataSamples, labels):
    for sample in dataSamples:

        for label in labels:
            if label['label'] == sample['label']:
                sample['label'] = label['vector']
    return dataSamples


