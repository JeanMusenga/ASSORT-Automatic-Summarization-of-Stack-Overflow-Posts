import math
from matplotlib.pyplot import axis
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
from torch.nn.functional import threshold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision.transforms as transforms
from torch.autograd import Variable
import re
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets,transforms
from nltk.stem import SnowballStemmer
import blitz
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
import torchbnn as bnn
import matplotlib.pyplot as plt

lr = pickle.load(open("lr_model.sav", "rb"))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(799, 450)
        self.fc3 = nn.Linear(450, 450)
        self.fc4 = nn.Linear(450, 15)
        self.fc5 = nn.Linear(15, 2)

    def forward(self, x):
        return self.fc5(self.fc4(self.fc3(F.relu(self.fc1(x)))))

def predict(net_inputs, threshold):
    prediction = []

    net = Net()
    net.load_state_dict(torch.load('1110model.pt'))
    if net_inputs != []:
        allresult = []
        for index, i in enumerate(net_inputs):
            result = net(torch.tensor(i, dtype=torch.float))
            result = nn.Softmax(dim=0)(result)
            # input(result)
            allresult.append((index, result[1]))
        # input(allresult)
        allresult.sort(key=lambda x: x[1], reverse=True)
        for i, j in allresult:
            if len(net_inputs) < 20 and (j > threshold or (len(prediction) < 4 and j > threshold - 0.1)):
                prediction.append((i, j))
            elif len(net_inputs) >=20 and len(prediction) < 5:
                prediction.append((i, j))

        if prediction == []:
            prediction.append(allresult[0])
            # elif j > threshold - 0.1 and len(prediction) * 3 < len(net_inputs):
            #     prediction.append(i)

    
    # if net_inputs != []:
    #     allresult = []
    #     for index, i in enumerate(net_inputs):
    #         result = lr.predict(np.array(i).reshape(1, -1))
    #         allresult.append((index, result[0]))
    #     # input(allresult)
    #     allresult.sort(key=lambda x: x[1], reverse=True)
    #     for i, j in allresult:
    #         if j > threshold and len(prediction) < 4:
    #             prediction.append((i, j))
        

    return prediction, prediction[0][0]