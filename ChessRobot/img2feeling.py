import cv2
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import PySimpleGUI as sg


class DeepEmotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 10, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(10, 10, 3)
        self.conv4 = nn.Conv2d(10, 10, 3)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.norm = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(810, 50)
        self.fc2 = nn.Linear(50, 7)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, input):
        out = self.stn(input)

        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3(out))
        out = self.norm(self.conv4(out))
        out = F.relu(self.pool4(out))

        out = F.dropout(out)
        out = out.view(-1, 810)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

class MLPNet(nn.Module):
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


def transform_img(img, transformation, device):
    img = Image.fromarray(img)
    img = transformation(img).float()
    img = torch.autograd.Variable(img,requires_grad = True)
    img = img.unsqueeze(0)
    return img.to(device)


def get_emotion_output(opt, img):
    classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
    opt = Opt()
    device = opt.device
    # pdb.set_trace()
    emotion_net = DeepEmotion()
    emotion_net.load_state_dict(torch.load(opt.emotion_model_path))
    emotion_net.to(device)
    transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(opt.frontalface)
    # Read the frame
    # _, img = vc.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    # pdb.set_trace()

    for (x, y, w, h) in faces:
        roi = img[y:y + h, x:x + w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (48, 48))
        imgg = transform_img(roi, transformation, device)
        out = emotion_net(imgg)
        # pdb.set_trace()
        pred = F.softmax(out)

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        classs = torch.argmax(pred, 1)
        wrong = torch.where(classs != 3, torch.tensor([1.]).cuda(), torch.tensor([0.]).cuda())
        classs = torch.argmax(pred, 1)
        prediction = classes[classs.item()]  # This is what we need for DDA!!!
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        pred_img = cv2.putText(img, prediction, org, font,
                          fontScale, color, thickness, cv2.LINE_AA)

        return True, pred_img, torch.squeeze(pred.cpu().detach()).numpy().tolist()
    return False, None, None


def get_feeling_output(opt, data):
    data = np.array(data)
    padded_data = data.copy()
    if len(data) == 0:
        return {'data': torch.tensor([0]), 'label': torch.tensor([0])}
        print("len is zero!")
    # print(os.path.join(self.path, file))
    for i in range(90 - len(data)):
        padding_data = data[random.randrange(0, len(data))].reshape(1, 7)
        padded_data = np.append(padded_data, padding_data, axis=0)

    data = padded_data
    data = torch.tensor(data).float()

    mlp_net = MLPNet()
    mlp_net.load_state_dict(torch.load(opt.mlp_model_path))
    outputs = mlp_net(data)
    _, predicted = torch.max(outputs.data, 1)
    return torch.squeeze(predicted.cpu()).item()



def ui():
    sg.theme('DarkAmber')  # Add a touch of color
    layout = [[sg.Text('Game begin', key='display_text')],
              [sg.CB('Start', key='start',  size=(12, 2)), sg.Button('done',  size=(12, 2))]]
    return layout


class Opt():
  def __init__(self):
    self.mlp_model_path = './Emotion2Feeling/model/latest_model.pth'
    self.mlp_net = MLPNet()
    self.mlp_net = self.mlp_net.load_state_dict(torch.load(self.mlp_model_path))
    self.emotion_model_path = './FacialExpressionRecognition/model/deep_emotion-500-128-0.005.pt'
    #self.emotion_net = DeepEmotion()
    #self.emotion_net = self.emotion_net.load_state_dict(torch.load(self.emotion_model_path))
    self.frontalface = './FacialExpressionRecognition/tools/haarcascade_frontalface_default.xml'
    # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # self.datasets = './Emotion2Feeling/data/datasets'


def main():
    opt = Opt()
    # mlp_net = MLPNet()
    # mlp_net.load_state_dict((torch.load(opt.mlp_model_path)))
    # emotion_net = DeepEmotion()
    # emotion_net.load_state_dict(torch.load(opt.emotion_model_path))

    layout = ui()
    window = sg.Window('CollectData', layout, resizable=True)

    vc = cv2.VideoCapture(0)
    last_time = 0
    pre_state = False

    with torch.no_grad():
        while True:
            _, img = vc.read()
            pred_img = img.copy()

            event, value = window.Read(timeout=100)

            cur_state = value['start']

            if pre_state == False and cur_state == True:
                results = []
            pre_state = cur_state

            if event == sg.WIN_CLOSED:  # if user closes window
                break

            if value['start']:
                result, tmp_img, output = get_emotion_output(opt, img)
                if result:
                    pred_img = tmp_img
                    if time.time() - last_time >= 2:
                        last_time = time.time()
                        if len(results) == 90:
                            del (results[0])
                        print("cur_time: ", time.time())
                        results.append(output)

            elif event == 'done':
                # label = 0 means too easy; label = 1 means too hard
                feeling = get_feeling_output(opt, results)
                if feeling == 0:
                    window['display_text'].update('Too easy')
                else:
                    window['display_text'].update('Too hard')
                results = []


            cv2.imshow('img', pred_img)

        window.close()





if __name__ == '__main__':
    main()