import os
import cv2
import pdb
import pandas as pd
import PySimpleGUI as sg
import torch
import torch.nn.functional as F
import multiprocessing
from torchvision import transforms
from PIL import Image

from FacialExpressionRecognition.model.network import DeepEmotion


class Opt():
  def __init__(self):
    # pdb.set_trace()
    self.model_path = '../model/deep_emotion-500-128-0.005.pt'
    self.frontalface = '../tools/haarcascade_frontalface_default.xml'
    self.result_path = '..results'
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.cam = True

def transform_img(img, transformation, device):
    # img = Image.open(path)
    # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = Image.fromarray(img)
    img = transformation(img).float()
    img = torch.autograd.Variable(img,requires_grad = True)
    img = img.unsqueeze(0)
    return img.to(device)


def ui():
    sg.theme('DarkAmber')  # Add a touch of color
    # All the stuff inside your window.
    layout = [
                    [sg.CB('Easy', key='easy'), sg.CB('Hard', key='hard')],
                    [sg.CB('Start', key='start'), sg.Button('end')],
                    ]

    return layout

def get_output(opt, vc):
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
    _, img = vc.read()
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
        return torch.squeeze(pred.cpu().detach()).numpy().tolist()

def save_results(results, results_path, idx):
    while (len(results) < 90):
        tmp = results[-1].copy()
        results.append(tmp)
    file_name = os.path.join(results_path, str(idx)) + '.txt'
    with open(file_name, 'w') as f:
        # pdb.set_trace()
        for row in results:
            f.writelines(str(row) + '\n')


def save_label(label, result_path, idx):
    file_name = os.path.join(result_path, 'label') + '.txt'
    with open(file_name, 'a') as f:
        f.write(str(idx) +  ' ' + str(label) + '\n')


def main():
    opt = Opt()
    layout = ui()
    window = sg.Window('CollectData', default_button_element_size=(12, 1), auto_size_buttons=False,
                       icon='interface_images/robot_icon.ico').Layout(layout)
    vc = cv2.VideoCapture(0)
    datasets = './datasets'
    numbers = len(os.listdir(datasets))
    idx = numbers if numbers >= 2 else 1
    results = []
    while True:
        button, value = window.Read(timeout=100)
        if button == sg.WIN_CLOSED or button == 'Cancel':  # if user closes window or clicks cancel
            break

        if value['start']:
            result_path = os.path.join('./datasets')
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            output = get_output(opt, vc)
            # pdb.set_trace()
            print(output)
            if len(results) == 90:
                del(results[0])
            if output:
                results.append(output)
        # if button == 'end':
        elif button == 'end':
            # label = 0 means too easy; label = 1 means too hard
            label = 0 if value['easy'] else 1
            save_results(results, result_path, idx)
            save_label(label, result_path, idx)
            results = []
            idx += 1

    window.close()

if __name__ == '__main__':
    main()