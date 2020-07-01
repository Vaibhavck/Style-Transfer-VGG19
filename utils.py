#imports
import re
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import requests
import torch
import torchvision


def load_image(image_path, shape=None, size=512):
    '''
    for loading image

    inputs : 
        1. image_path - path or link to the image
        2. shape - shape of the image (optional)
        3. size - max size of the image (optional)

    returns : 
        image in form of pytorch tensor

    '''

    if image_path.startswith('http'):
        image = Image.open(BytesIO(requests.get(image_path).content))
    else:
        image =  Image.open(image_path)

    image = image.convert('RGB')
    image_size = size if max(image.size) > size else max(image.size)
    if shape is not None:
        image_size = shape
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transforms(image)[:3, :, :].unsqueeze(0)

def convert_tensor(tensor):
    '''
    conversion for displaying image

    inputs : 
        image tensor

    returns : 
        displayable image

    '''
    
    tensor = tensor.to("cpu").clone().detach()
    tensor = tensor.numpy().squeeze()
    tensor = tensor.transpose(1,2,0)
    tensor = tensor * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    tensor = tensor.clip(0, 1)
    return tensor

# get features
def extract_featrues(image, model):
    '''
    for extracting features from image

    inputs : 
        1. image - image in form  of pytorch tensor
        2. model - model to extract features from image

    returns : 
        feature dict

    '''

    layers = {
        '0' : 'conv1_1',
        '5' : 'conv2_1',
        '10' : 'conv3_1',
        '19' : 'conv4_1',
        '21' : 'conv4_2',    # for content
        '28' : 'conv5_1',
    }
    features = {}
    for layer_id, layer in model._modules.items():
        image = layer(image)
        if layer_id in layers:
            features[layers[layer_id]] = image

    return features

# gram matrix
def gram_matrix(tensor):
    '''
    gram matrix

    inputs : 
        1. tensor - tensor to get its gram matrix

    returns : 
        gram matrix of the tensor

    '''

    _, num_channels, height, width = tensor.size()
    tensor = tensor.view(num_channels, height*width)
    return torch.mm(tensor, tensor.t())
