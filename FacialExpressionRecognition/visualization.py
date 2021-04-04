import os
import sys
import cv2
import pdb
import torch
import torch.nn.functional as F
import multiprocessing
import chessBotConst as cbConst
from torchvision import transforms
from PIL import Image
from time import time


#file_root = os.getcwd()
# file_root = sys.path.append(os.getcwd())

# pdb.set_trace()

from FacialExpressionRecognition.model.network import DeepEmotion
#from model.network import DeepEmotion

class Opt():
  def __init__(self):
    # pdb.set_trace()
    self.model_path = './FacialExpressionRecognition/model/deep_emotion-500-128-0.005.pt'
    self.frontalface = './FacialExpressionRecognition/tools/haarcascade_frontalface_default.xml'
    self.result_path = './FacialExpressionRecognition/results'
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.cam = True



def load_img(path, transformation, device):
    img = Image.open(path)
    img = transformation(img).float()
    img = torch.autograd.Variable(img,requires_grad = True)
    img = img.unsqueeze(0)
    return img.to(device)


def start_facialExpress_recog_mod(msgQueue, ctrlQueue):
    print("enter facialExpress recog!")
    opt = Opt()
    device = opt.device
    # pdb.set_trace()
    net = DeepEmotion()
    net.load_state_dict(torch.load(opt.model_path))
    net.to(device)
    classes = ('Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral')
    transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    if opt.cam:
        # Load the cascade
        face_cascade = cv2.CascadeClassifier(opt.frontalface)

        # To capture video from webcam.
        cap = cv2.VideoCapture(1)
        
        lastPredictTime = time();
        while True:
            
            # Read the frame
            _, img = cap.read()
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect the faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # Draw the rectangle around each face
            #pdb.set_trace()
            for (x, y, w, h) in faces:
                roi = img[y:y+h, x:x+w]
                roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                roi = cv2.resize(roi,(48,48))
                # cv2.imwrite(r"F:\Projects\Deep-Emotion-master\imgs/roi.jpg", roi)
                # pdb.set_trace()
                cv2.imwrite(os.path.join(opt.result_path, 'roi.jpg'), roi)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # imgg = load_img(r"F:\Projects\Deep-Emotion-master\imgs/roi.jpg")
            imgg = load_img(os.path.join(opt.result_path, 'roi.jpg'), transformation, device)
            out = net(imgg)
            # pdb.set_trace()
            pred = F.softmax(out)
            classs = torch.argmax(pred,1)
            wrong = torch.where(classs != 3,torch.tensor([1.]).cuda(),torch.tensor([0.]).cuda())
            classs = torch.argmax(pred,1)
            prediction = classes[classs.item()] #This is what we need for DDA!!!
            if ctrlQueue != None and ctrlQueue.empty() == False:#process communication with chess robot, if ctrlQueue not empty write prdection to anotherQueue
                curTime = time()
                if curTime - lastPredictTime > cbConst.SampleFacialExpressionTimeGap:
                    print("write to msgQueue")
                    if msgQueue.full():
                        msgQueue.get()
                    toTransMsg = torch.squeeze(pred.cpu().detach()).numpy().tolist()
                    msgQueue.put(toTransMsg)
                    print(toTransMsg)
                    lastPredictTime = curTime
                    
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            img = cv2.putText(img, prediction, org, font,
                           fontScale, color, thickness, cv2.LINE_AA)

            cv2.imshow('img', img)
            # Stop if (Q) key is pressed
            k = cv2.waitKey(30)
            if k==ord("q"):
                break

        # Release the VideoCapture object
        cap.release()
#start_facialExpress_recog_mod(None, None)