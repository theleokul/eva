import torch
import torchvision as tv
import numpy as np

MODEL_DIR = "model_weights_25.pth"

device = torch.device('cuda')

CLASS2EMOTION = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'neutral',
    5: 'sadness',
    6: 'surprise(negative)',
    7: 'surprise(positive)',
}

def add_stats_to_aggregation(aggregation, stats):
  for k in stats.keys():
    aggregation[k] += stats[k]
  return aggregation

def load_model(dir = MODEL_DIR, device=device):
  model = tv.models.resnet18(pretrained=False)
  model.fc = torch.nn.Linear(512, 8)
  model = model.to(device)
  model.load_state_dict(torch.load(dir,map_location=torch.device('cpu')))
  model.eval()
  return model

import torch.nn.functional as F

def apply_model(model, npimg, device):
  # assert npimg.shape == (224, 224, 3)
  npimg = ((npimg / 255) - 0.5) * 2
  input = np.expand_dims(np.moveaxis(npimg, -1, 0), 0)
  input = torch.tensor(input)
  model = model.float()
  model = model.to(device)
  input = input.to(device)
  output = model(input.float())
  logits = torch.nn.Softmax()(output)
  emotions_dict = {}
  # print(logits.detach().numpy().tolist())
  for i, l in enumerate(logits.detach().cpu().numpy().tolist()[0]):
    emotions_dict[CLASS2EMOTION[i]] = l
  return emotions_dict

import cv2

def proccess_video(model, video, device = device,):
  cap = cv2.VideoCapture(video)
  ret, frame = cap.read()
  counter = 0
  aggregation = {}
  for k in CLASS2EMOTION.keys():
    aggregation[CLASS2EMOTION[k]] = 0

  while(1):
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
      cap.release()
      cv2.destroyAllWindows()
      break
    # print(frame.shape)
    # h, w = frame.shape[:2]
    # print(h, w)
    # frame = cv2.resize(frame, (w // 2, h // 2))
    # print(frame.shape)
    statistics = apply_model(model, frame, device)
    add_stats_to_aggregation(aggregation, statistics)
    counter += 1

  for k in aggregation.keys():
    aggregation[k] /= counter
  return aggregation