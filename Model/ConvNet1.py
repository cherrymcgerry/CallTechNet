import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

USE_SIGMOID = False
#convolution is de filter die naar lokale regio kijkt en dan dot product neemt van de 2 vectors (filtervector en lokaleregio vector) en output wordt dan 1 (pixel) van u activation map. maxPooling is ipv dotproduct pakje gwn grootste value. pooling layers pakje meestal stride waarbij da filter nie overlapt.
#maar om de zoveel layers ne keer pooling doen. bij pooling layers wordt geen zeropadding gebruikt en meestal filter size 2 of 3 en stride 2
#max pooling is beter omdaje eeen signaal krijgt van hoogste firing. en das eigenlijk logisch want ge wilt weten waardatie t meest activeerd
#ge kunt stride gebuiken ipv pooling. en meer en meer wordt enkel stride gebruikt vo te downsizen. geeft ook betere results.
#zeropad data!!! for corners common : (filtersize -1)/2
#stride is lik pooling omdaje ook u output resolutie verkleint. werkt soms beter dan pooling layers. verkleint de size van de actrivation maps per layer en dan hebje ook minder parameters.
# (input - filter) / stride + 1 = output (moet altijd int zijn want decimal duidt op filter die niet overal kan toegepast worden)
#fixed learning rate??? opzoeken
#toepassen data augmentation om dataset te vergroten en overfitting verminderen.

#conv layers en pool layers ebben verschillende settings.

#fully connected layers. nemen conv output en stretch them out. dan verkleinen totdaje u score outputs ebt.
#trend nu is weggaan van pool en fully connected layers en enkel convlayers gebruiken

# TODO add Downconv module te clean up code


class ConvNet(nn.Module):

    def __init__(self):
        x = 32
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.conv2 = nn.Conv2d(128, 256, 3)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 3)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 3)
        self.conv4_bn = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv5_bn = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(128, 256, 3)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3)
        self.conv7_bn = nn.BatchNorm2d(512)

        self.layer1 = self.conv1
        self.layer2 = nn.Sequential()
        self.layer2.add_module("Conv2", self.conv2)
        self.layer2.add_module("BN2", self.conv2_bn)

        self.layer3 = nn.Sequential()
        self.layer3.add_module("Conv3", self.conv3)
        self.layer3.add_module("BN3", self.conv3_bn)

        self.layer4 = nn.Sequential()
        self.layer4.add_module("Conv4", self.conv4)
        self.layer4.add_module("BN4", self.conv4_bn)

        self.layer5 = nn.Sequential()
        self.layer5.add_module("Conv5", self.conv5)
        self.layer5.add_module("BN5", self.conv5_bn)

        self.layer6 = nn.Sequential()
        self.layer6.add_module("Conv6", self.conv6)
        self.layer6.add_module("BN6", self.conv6_bn)

        self.layer7 = nn.Sequential()
        self.layer7.add_module("Conv7", self.conv7)
        self.layer7.add_module("BN7", self.conv7_bn)

        x = torch.randn(128, 128).view(-1, 1, 128, 128)
        self._to_linear = None
        self.dropout = nn.Dropout2d(0.5)
        self.convs(x)



        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc1_bn = nn.BatchNorm2d(1024)
        self.fc2 = nn.Linear(self._to_linear, 512)
        self.fc2_bn = nn.BatchNorm2d(512)
        self.fc3 = nn.Linear(512, 101)

    def convs(self, x):

        if USE_SIGMOID:
            x = F.max_pool2d(F.sigmoid(self.layer1(x)), (2, 2))
            x = self.dropout(x)
            x = F.max_pool2d(F.sigmoid(self.layer2(x)), (2, 2))
            x = self.dropout(x)
            x = F.max_pool2d(F.sigmoid(self.layer3(x)), (2, 2))
            x = self.dropout(x)
            x = F.max_pool2d(F.sigmoid(self.layer4(x)), (2, 2))
            #x = self.dropout(x)
            #x = F.max_pool2d(F.sigmoid(self.layer5(x)), (2, 2))
            #x = F.max_pool2d(F.sigmoid(self.layer6(x)), (2, 2))
            #x = F.max_pool2d(F.sigmoid(self.layer7(x)), (2, 2))

        else:
            x = F.max_pool2d(F.leaky_relu(self.layer1(x)), (2, 2))
            #x = self.dropout(x)
            x = F.max_pool2d(F.leaky_relu(self.layer2(x)), (2, 2))
            #x = self.dropout(x)
            x = F.max_pool2d(F.leaky_relu(self.layer3(x)), (2, 2))
            #x = self.dropout(x)
            x = F.max_pool2d(F.leaky_relu(self.layer4(x)), (2, 2))

            #x = F.relu(self.layer5(x))
            #x = F.relu(self.layer6(x))
            #x = F.max_pool2d(F.relu(self.layer7(x)), (2, 2))
            x = self.dropout(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        # x = F.sigmoid(self.fc1_bn(self.fc1(x)))
        #x = F.sigmoid(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)


for i in range(1, num_fc):
    nfInput = self.to_linear
    self.to_linear = int(self.to_linear / 2)
    submodule = FullyConnected(nfInput, self.to_linear, submodule)

submodule = FullyConnected(self.to_linear, 2, submodule, final=True)