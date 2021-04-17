import os
import numpy as np
import math
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from PIL import Image
import io

def predict_new_image(imgoriginal, model, threshold=0.7, device='cpu'):
    img = imgoriginal.copy()
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.to('cpu')
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    model.eval()

    # forward pass
    out = model.forward(batch_t)
    train_data_path = "C:/Users/alejg/Documents/proyectos_personales/image_classification/dogs1GB/dogImages/dogImages/train/"

    classes = os.listdir(train_data_path)
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    jsonOutput = {}
    confianza = percentage[index[0]].item()
    print(percentage[index[0]].item())
    print(classes[index[0]])
    if confianza > threshold:
        jsonOutput.update({
            'confidence': confianza,
            'label': classes[index[0]]
        })
    return jsonOutput
