import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self):
        self.path = './data/datasets/train'
        self.data_list = os.listdir('./data/datasets/train')
        self.label = './data/datasets/label/label.csv'


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        file = self.data_list[index]
        data = pd.read_csv(os.path.join(self.path, file), index_col=False).astype('float')
        data = np.array(data)
        data = torch.tensor(data).float()
        label = pd.read_csv(self.label).astype('float')
        label = label['label'][index]
        label = np.array([1, 0]) if label == 0 else np.array([0, 1])
        label = torch.tensor(label).float()
        sample = {'data': data, 'label': label}
        return sample

    def __len__(self):
        return len(self.data_list)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(720, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.view(-1, 720)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = CustomDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    model = Net().to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):
        running_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data['data'].squeeze().cuda(), data['label'].cuda()
            # print(inputs.size())
            # print(labels)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                torch.save(model.state_dict(), './model/latest_model.pth')

    print('Finished Training')

def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Net().to(device)
    print(model)

if __name__ == '__main__':
    train()
    # test()