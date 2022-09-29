import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class IrisDataset(Dataset):
    def __init__(self):
        xy_train = np.loadtxt('./iris/iris-test.txt', delimiter=" ", dtype=np.float32)
        self.x_train = torch.from_numpy(xy_train[:, 1:])
        self.y_train = torch.from_numpy(xy_train[:, [0]])
        self.n_samples = xy_train.shape[0]
        self.n_features = xy_train.shape[1] - 1
        
        """
        xy_test = np.loadtxt('./iris/iris-test.txt', delimiter=" ", dtype=np.float32)
        self.x_test = torch.from_numpy(xy_test[:, 1:])
        self.y_test = torch.from_numpy(xy_test[:, [0]])
        """
        
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]
    
    def __len__(self):
        return self.n_samples


class SoftmaxClassify(nn.Module):
    
    def __init__(self):
        super(SoftmaxClassify, self).__init__()
        self.linear1 = nn.Linear(2, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        #x = self.softmax(x)
        x = self.linear3(x)
        
        return x
    
#dataset = IrisDataset()
#dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

train_Xy =  np.loadtxt('./iris/iris-test.txt', delimiter=" ", dtype=np.float32)
np.random.shuffle(train_Xy)
train_X = train_Xy[:, 1:]
train_y = train_Xy[:, [0]]
train_X = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_y)
train_y = train_y[:, 0].long()
train_y = train_y.add(-1)

test_Xy = np.loadtxt('./iris/iris-test.txt', delimiter=" ", dtype=np.float32)
test_X = test_Xy[:, 1:]
test_y = test_Xy[:, [0]]
test_X = torch.from_numpy(test_X)
test_y = torch.from_numpy(test_y)
test_y = test_y[:, 0].long()
test_y = test_y.add(-1)


model = SoftmaxClassify()

criterion = nn.CrossEntropyLoss()

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 10000

for epoch in range(num_epochs):
    optimizer.zero_grad()
        
    out = model(train_X)
    #out = torch.tensor(out, requires_grad=True)
    #print(out.size())
        
    loss = criterion(out, train_y)
    
    loss.backward()
    optimizer.step()

    
    if ((epoch + 1) % 10 == 0):
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')
        
with torch.no_grad():
    y_predicted = model(test_X)
    y_predicted_cls = y_predicted.round()
    acc = torch.sum(torch.softmax(y_predicted_cls, dim=1).argmax(dim=1) == test_y)
    acc = acc / float(test_y.shape[0])
    print(f'accuracy = {acc:.4f}')