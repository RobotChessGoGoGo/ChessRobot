import os
import cv2
import pdb
import time
import pandas as pd
import PySimpleGUI as sg
import torch
import torch.nn.functional as F
import multiprocessing
from torchvision import transforms
from PIL import Image

from FacialExpressionRecognition.model.network import DeepEmotion
import os

class Opt():
  def __init__(self):
    self.model_path = './FacialExpressionRecognition/model/deep_emotion-500-128-0.005.pt'
    self.frontalface = './FacialExpressionRecognition/tools/haarcascade_frontalface_default.xml'
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.datasets = './Emotion2Feeling/data/datasets'

def transform_img(img, transformation, device):
    img = Image.fromarray(img)
    img = transformation(img).float()
    img = torch.autograd.Variable(img,requires_grad = True)
    img = img.unsqueeze(0)
    return img.to(device)

# [sg.CB('Easy', key='easy'), sg.CB('Hard', key='hard')]

def ui():
    sg.theme('DarkAmber')  # Add a touch of color
    layout = [[sg.Radio('Easy', key='easy', group_id=1, size=(12, 1), default=True), sg.Radio('Hard', key='hard', group_id=1,  size=(12, 1))],
              [sg.CB('Start', key='start',  size=(12, 2)), sg.Button('save',  size=(12, 2))]]
    return layout

def get_output(opt, img):
    classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
    opt = Opt()
    device = opt.device
    # pdb.set_trace()
    net = DeepEmotion()
    net.load_state_dict(torch.load(opt.model_path))
    net.to(device)
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
        out = net(imgg)
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

def save_results_csv(results, results_path, idx):
    while len(results) < 90 :
        results.append(results[-1])
    data = pd.DataFrame(results)
    data.to_csv(os.path.join(results_path, str(idx)) + '.csv')

def save_label_csv(label, result_path, idx):
    content = [[idx, label]]
    data = pd.DataFrame(content, columns=['idx', 'label'])
    data.to_csv(os.path.join(result_path, 'label') + '.csv', mode='a', header=True if idx == 1 else False, index=False)

def main():
    global classes
    classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
    opt = Opt()
    datasets_folder = opt.datasets
    if not os.path.exists(datasets_folder):
        os.makedirs(datasets_folder)

    train_folder = os.path.join(datasets_folder, 'train')
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    numbers = len(os.listdir(train_folder))

    label_folder = os.path.join(datasets_folder, 'label')
    if not os.path.exists(label_folder):
        os.makedirs((label_folder))

    idx = numbers + 1 if numbers >= 1 else 1
    layout = ui()
    window = sg.Window('CollectData', layout, resizable=True)
    results = []
    vc = cv2.VideoCapture(0)
    last_time = 0

    while True:
        _, img = vc.read()
        pred_img = img.copy()

        button, value = window.Read(timeout=100)
        if button == sg.WIN_CLOSED:  # if user closes window
            break

        if value['start']:
            result, tmp_img, output = get_output(opt, img)
            if result:
                pred_img = tmp_img
                if time.time() - last_time >= 2:
                    last_time = time.time()
                    if len(results) == 90:
                        del(results[0])
                    print("cur_time: ", time.time())
                    results.append(output)

        elif button == 'save':
            # label = 0 means too easy; label = 1 means too hard
            label = 0 if value['easy'] else 1
            save_results_csv(results, train_folder, idx)
            save_label_csv(label, label_folder, idx)
            results = []
            idx += 1

        cv2.imshow('img', pred_img)

    window.close()

if __name__ == '__main__':
    main()