# imports
import re
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import utils
import requests
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler

# using vgg19 pretrained model
vgg_model = torchvision.models.vgg19(pretrained=True).features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg_model.to(device)
# print(vgg_model)

# loading content and style images
content_img = utils.load_image('https://petpress.net/wp-content/uploads/2019/11/famous-white-horse-names.jpg').to(device)
style_image = utils.load_image('https://5.imimg.com/data5/FW/TM/MY-2420068/designer-abstract-painting-500x500.jpg').to(device)

# extracting features
content_features = utils.extract_featrues(content_img, vgg_model)
style_features = utils.extract_featrues(style_image, vgg_model)

# gram matrix for style image
style_grams = {layer:utils.gram_matrix(style_features[layer]) for layer in style_features}

# creating target image by cloning content image
target_image = content_img.clone().requires_grad_(True).to(device)

# specify weights (importance) for each layer
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

# weigths for content and style losses
alpha = 5
beta = 1000 


# training
optimizer = torch.optim.Adam([target_image])
num_iters = 10000

for i in range(1, num_iters+1):
    print(f'epoch {i} / {num_iters}')
    target_features = utils.extract_featrues(target_image, vgg_model)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram_mat = utils.gram_matrix(target_feature)
        _, num_channels, height, width = target_feature.shape
        style_gram_mat = style_grams[layer]
        l_style_loss = style_weights[layer] * torch.mean((target_gram_mat - style_gram_mat)**2)
        style_loss += l_style_loss / (num_channels*height*width)

    total_loss = alpha * content_loss + beta * style_loss

    optimizer.zero_grad()
    total_loss.backward(retain_graph=True)
    optimizer.step()

    if i % 500 == 0:
        print('current loss', total_loss.item())
        plt.imsave(f'samples/target_at_{i}.png', utils.convert(target_image))