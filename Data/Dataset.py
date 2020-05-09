import pickle
import os
import torch
from skimage import io, transform
from torchvision.transforms import transforms
from PIL import Image
import numpy as np

from torch.utils.data import Dataset

OUTPUT_SIZE = (64, 64)

class data(Dataset):

    def __init__(self, dataroot, is_train, device='cpu', normalize=True):
        path = os.path.join(dataroot, 'train_dataColor.data' if is_train else 'val_dataColor.data')
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        if path == '../101_ObjectCategories\\train_dataColor.data':
            with open(os.path.join(dataroot,"test_dataColor.data"), 'rb') as tf:
                self.data.extend(pickle.load(tf))

        #self.normalize = normalize
        self.device = device
        self.onehot = np.eye(101)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fileLocation = self.data[idx]['image']
        inputI = Image.open(open(fileLocation, 'rb')).convert('RGB')
        imageShow = inputI
        imagelabel = self.data[idx]['label']
        #output = self.data[idx]['label']

        #inputI = rescale(inputI, OUTPUT_SIZE)
        #inputI = torch.from_numpy(inputI)
        #inputI = inputI.view(1, 64, 64)

        #if self.normalize:
        composed = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
        inputI = composed(inputI)

        output = self.onehot[self.data[idx]['label']]

        #imageShow.show()

        return inputI, output

