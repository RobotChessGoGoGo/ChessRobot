import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self):
        self.path = './data/datasets/train'
        self.data_list = os.listdir('./data/datasets/train')
        self.data_list.sort()
        self.label = './data/datasets/label/label.csv'
        # self.model_path = '/model/latest_model.pth'


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        file = self.data_list[index]
        file_idx = int(file[: -4])
        data = pd.read_csv(os.path.join(self.path, file), index_col=0).astype('float')
        data = np.array(data)

        ## Padding data
        padded_data = data.copy()
        # print(len(data))
        if len(data) == 0:
            return {'data': torch.tensor([0]), 'label': torch.tensor([0])}
            print("len is zero!")
        # print(os.path.join(self.path, file))
        for i in range(90 - len(data)):
            padding_data = data[random.randrange(0, len(data))].reshape(1, 7)
            padded_data = np.append(padded_data, padding_data, axis=0)

        data = padded_data
        data = torch.tensor(data).float()
        label = pd.read_csv(self.label).astype('float')
        # print("file: ", file, "label_idx: ", label['idx'][file_idx - 1])
        label = label['label'][file_idx - 1]
        # label = np.array([1, 0]) if label == 0 else np.array([0, 1])
        label = torch.tensor(label).float()
        sample = {'data': data, 'label': label}
        return sample

    def __len__(self):
        return len(self.data_list)

class TestDataset(Dataset):
    def __init__(self):
        self.path = './data/datasets/test'
        self.data_list = os.listdir('./data/datasets/test')
        self.data_list.sort()
        self.label = './data/datasets/label/label.csv'
        # self.model_path = '/model/latest_model.pth'


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        file = self.data_list[index]
        file_idx = int(file[: -4])
        data = pd.read_csv(os.path.join(self.path, file), index_col=0).astype('float')
        data = np.array(data)

        ## Padding data
        padded_data = data.copy()
        # print(len(data))
        if len(data) == 0:
            return {'data': torch.tensor([0]), 'label': torch.tensor([0])}
            print("len is zero!")
        # print(os.path.join(self.path, file))
        for i in range(90 - len(data)):
            padding_data = data[random.randrange(0, len(data))].reshape(1, 7)
            padded_data = np.append(padded_data, padding_data, axis=0)

        data = padded_data
        data = torch.tensor(data).float()
        label = pd.read_csv(self.label).astype('float')
        # print("file: ", file, "label_idx: ", label['idx'][file_idx - 1])
        label = label['label'][file_idx - 1]
        # label = np.array([1, 0]) if label == 0 else np.array([0, 1])
        label = torch.tensor(label).float()
        sample = {'data': data, 'label': label}
        return sample

    def __len__(self):
        return len(self.data_list)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(630, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5= nn.Linear(128, 2)

    def forward(self, x):
        x = x.view(-1, 630)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # x = F.softmax(self.fc5(x), dim=1)
        x = self.fc5(x)
        return x


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = CustomDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    model = Net().to(device)
    print(model)
    model_path = './model/latest_model.pth'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(1000):
        running_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data['data'].squeeze().cuda(), data['label'].cuda()
            if inputs.all() == torch.tensor([0]).cuda():
                continue
            # print(inputs.size())
            # print(labels)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                # torch.save(model.state_dict(), './model/latest_model.pth')

            if epoch % 5 == 4:
                torch.save(model.state_dict(), model_path)

    print('Finished Training')

def test():
    model_path = './model/latest_model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = TestDataset()
    testloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    model = Net()
    model.load_state_dict(torch.load(model_path))
    print(model)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data['data'].squeeze(), data['label']
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += 1 if predicted == labels else 0

    print("Accuracy of the network on ", total, " samples: ", 100 * correct/total, " %")


if __name__ == '__main__':
    # train()
    test()