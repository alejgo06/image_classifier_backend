from fastapi import FastAPI
from typing import List
from fastapi import FastAPI, UploadFile, File
import numpy as np
from starlette.requests import Request
import io
from PIL import Image
import base64

from utils import predict_new_image
import torch
import os
from torchvision import datasets, transforms, models
import torch.nn as nn
import cv2
from collections import OrderedDict
app = FastAPI(title="Image classifier",
    description="Api",
    version="21.02.22.18.12")





@app.post("/predictDOGS")
async def analyseDOGS(image_file_read: bytes = File(...)):
    model = models.vgg16(pretrained=True)

    classifier = nn.Sequential(OrderedDict([

        ('fc1', nn.Linear(25088, 500)),
        ('relu', nn.ReLU()),
        ('drop1', nn.Dropout(p=0.95)),
        ('fc2', nn.Linear(500, 133)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model.load_state_dict(torch.load(os.getcwd() + "/models/generator_124.pth", map_location=torch.device('cpu')))
    file = base64.b64encode(image_file_read)
    jpg_original = base64.b64decode(file)
    # jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    # original_image = cv2.imdecode(jpg_as_np, flags=1)
    image = Image.open(io.BytesIO(jpg_original))
    model.to('cpu')
    model.eval()
    prediction=predict_new_image(image, model, threshold=0.1, device='cpu')
    return prediction
